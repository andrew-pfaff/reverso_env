"""Training entrypoint for the UCI HAR Reverso baseline."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from reverso.data import UCIHARReversoDataset, apply_batch_mixup
from reverso.models import build_model, validate_backend_requirements

from .config import TrainingConfig
from .metrics import ForecastMetricAccumulator
from .visualization import save_run_artifacts


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_subset(dataset: Dataset[Any], max_windows: int | None, seed: int) -> Dataset[Any]:
    """Optionally cap a dataset to a deterministic random subset."""
    if max_windows is None or max_windows >= len(dataset):
        return dataset
    if max_windows <= 0:
        raise ValueError(f"max_windows must be positive when provided, got {max_windows}.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_windows].tolist()
    return Subset(dataset, indices)


def build_dataloaders(config: TrainingConfig) -> dict[str, DataLoader[Any]]:
    """Build train/val/test dataloaders from the shared dataset contract."""
    input_channels, target_channels = config.resolved_channels()
    data_path = Path(config.data_path).expanduser() if config.data_path is not None else None
    device = config.resolved_device()
    pin_memory = device.type == "cuda"
    augmentation = config.make_augmentation_config()

    train_dataset = UCIHARReversoDataset(
        split="train",
        context_len=config.context_len,
        pred_len=config.pred_len,
        input_channels=input_channels,
        target_channels=target_channels,
        data_path=data_path,
        activity_subset=config.activity_subset,
        augmentation=augmentation,
    )
    val_dataset = UCIHARReversoDataset(
        split="val",
        context_len=config.context_len,
        pred_len=config.pred_len,
        input_channels=input_channels,
        target_channels=target_channels,
        data_path=data_path,
        activity_subset=config.activity_subset,
    )

    datasets: dict[str, Dataset[Any]] = {
        "train": _make_subset(train_dataset, config.train_max_windows, config.seed),
        "val": _make_subset(val_dataset, config.val_max_windows, config.seed + 1),
    }

    if config.eval_test:
        test_dataset = UCIHARReversoDataset(
            split="test",
            context_len=config.context_len,
            pred_len=config.pred_len,
            input_channels=input_channels,
            target_channels=target_channels,
            data_path=data_path,
            activity_subset=config.activity_subset,
        )
        datasets["test"] = _make_subset(test_dataset, config.test_max_windows, config.seed + 2)

    return {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            pin_memory=pin_memory,
        )
        for split, dataset in datasets.items()
    }


def _extract_batch(batch: dict[str, Tensor], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Move the model-facing parts of a batch to the requested device."""
    context = batch["context"].to(device=device, non_blocking=True)
    target = batch["target"].to(device=device, non_blocking=True)
    labels = batch["label"]
    return context, target, labels


def _mean_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Baseline Reverso training objective."""
    return nn.functional.l1_loss(prediction, target)


def _autocast_context(
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
):
    """Create an autocast context when mixed precision is enabled."""
    if not amp_enabled:
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError("AMP is only supported for CUDA runs in this training entrypoint.")
    if amp_dtype is None:
        raise RuntimeError("AMP was enabled, but no autocast dtype was resolved.")
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _validate_runtime(
    config: TrainingConfig,
    *,
    model_config,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> None:
    """Validate runtime/backend compatibility before starting a run."""
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but CUDA is not available.")
    if config.amp and device.type != "cuda":
        raise RuntimeError("AMP is only supported for CUDA runs in this training entrypoint.")
    if device.type == "cuda" and config.amp and amp_dtype == torch.bfloat16:
        supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not supports_bf16:
            raise RuntimeError("dtype='bf16' was requested, but this CUDA device does not support bfloat16.")
    validate_backend_requirements(
        model_config,
        device=device,
        amp_enabled=config.amp,
        amp_dtype=amp_dtype,
    )


def _ensure_finite_batch(
    *,
    split: str,
    batch_index: int,
    context: Tensor,
    target: Tensor,
    prediction: Tensor,
    loss: Tensor,
) -> None:
    """Fail fast when the model produces non-finite values."""
    tensors = {
        "context": context,
        "target": target,
        "prediction": prediction,
    }
    bad_tensors = [name for name, tensor in tensors.items() if not torch.isfinite(tensor).all()]
    if not bad_tensors and torch.isfinite(loss).all():
        return

    context_max = float(context.detach().abs().max().item())
    target_max = float(target.detach().abs().max().item())
    prediction_finite = bool(torch.isfinite(prediction).all().item())
    raise RuntimeError(
        f"Non-finite values detected during {split} on batch {batch_index}. "
        f"bad_tensors={bad_tensors or 'loss_only'} "
        f"context_abs_max={context_max:.6f} "
        f"target_abs_max={target_max:.6f} "
        f"prediction_finite={prediction_finite}"
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    target_channels: tuple[str, ...] | list[str],
    augmentation_config,
    grad_clip_norm: float | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    grad_scaler: GradScaler | None = None,
) -> dict[str, float]:
    """Train for one epoch and return summarized metrics."""
    model.train()
    metric_accumulator = ForecastMetricAccumulator(target_channels)
    total_loss = 0.0
    total_points = 0

    for batch_index, batch in enumerate(dataloader):
        context, target, labels = _extract_batch(batch, device)
        if augmentation_config is not None:
            context, target, labels = apply_batch_mixup(
                context,
                target,
                labels,
                config=augmentation_config,
            )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
            prediction = model(context)
            loss = _mean_loss(prediction, target)
        _ensure_finite_batch(
            split="train",
            batch_index=batch_index,
            context=context,
            target=target,
            prediction=prediction,
            loss=loss,
        )
        if grad_scaler is not None and grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0.0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        forecast_points = target.numel()
        total_loss += float(loss.item()) * forecast_points
        total_points += int(forecast_points)
        metric_accumulator.update(prediction, target, labels)

    metrics = metric_accumulator.compute()
    metrics["loss"] = total_loss / total_points
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader[Any],
    *,
    device: torch.device,
    target_channels: tuple[str, ...] | list[str],
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Evaluate the model on a dataloader."""
    model.eval()
    metric_accumulator = ForecastMetricAccumulator(target_channels)
    total_loss = 0.0
    total_points = 0

    for batch_index, batch in enumerate(dataloader):
        context, target, labels = _extract_batch(batch, device)
        with _autocast_context(device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
            prediction = model(context)
            loss = _mean_loss(prediction, target)
        _ensure_finite_batch(
            split="eval",
            batch_index=batch_index,
            context=context,
            target=target,
            prediction=prediction,
            loss=loss,
        )

        forecast_points = target.numel()
        total_loss += float(loss.item()) * forecast_points
        total_points += int(forecast_points)
        metric_accumulator.update(prediction, target, labels)

    metrics = metric_accumulator.compute()
    metrics["loss"] = total_loss / total_points
    return metrics


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Prefix split-specific metric names for logging."""
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def run_training(config: TrainingConfig) -> dict[str, Any]:
    """Run training and return the trained model and logged history."""
    set_seed(config.seed)
    device = config.resolved_device()
    dataloaders = build_dataloaders(config)
    amp_dtype = config.resolved_amp_dtype()
    model_config = config.make_model_config()
    _validate_runtime(config, model_config=model_config, device=device, amp_dtype=amp_dtype)
    model = build_model(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    grad_scaler = None
    if config.amp and amp_dtype == torch.float16:
        grad_scaler = GradScaler(device="cuda")

    history: list[dict[str, float]] = []
    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            device=device,
            target_channels=model_config.target_channels,
            augmentation_config=config.make_augmentation_config(),
            grad_clip_norm=config.grad_clip_norm,
            amp_enabled=config.amp,
            amp_dtype=amp_dtype,
            grad_scaler=grad_scaler,
        )
        val_metrics = evaluate(
            model,
            dataloaders["val"],
            device=device,
            target_channels=model_config.target_channels,
            amp_enabled=config.amp,
            amp_dtype=amp_dtype,
        )
        epoch_summary = {
            "epoch": float(epoch),
            **_prefix_metrics("train", train_metrics),
            **_prefix_metrics("val", val_metrics),
        }
        history.append(epoch_summary)
        print(json.dumps(epoch_summary, sort_keys=True))

    test_metrics = None
    if "test" in dataloaders:
        test_metrics = evaluate(
            model,
            dataloaders["test"],
            device=device,
            target_channels=model_config.target_channels,
            amp_enabled=config.amp,
            amp_dtype=amp_dtype,
        )
        print(json.dumps(_prefix_metrics("test", test_metrics), sort_keys=True))

    artifact_paths = None
    output_dir = config.resolved_output_dir()
    if output_dir is not None:
        artifact_paths = save_run_artifacts(
            output_dir,
            config=config,
            history=history,
            test_metrics=test_metrics,
            model=model,
            dataloaders=dataloaders,
            device=device,
            input_channels=model_config.input_channels,
            target_channels=model_config.target_channels,
            amp_enabled=config.amp,
            amp_dtype=amp_dtype,
        )

    return {
        "model": model,
        "config": config,
        "history": history,
        "test_metrics": test_metrics,
        "artifact_paths": artifact_paths,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for baseline training runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="all_to_all")
    parser.add_argument("--activity-subset", default="all")
    parser.add_argument("--context-len", type=int, default=384)
    parser.add_argument("--pred-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--long-conv-backend", default="native")
    parser.add_argument("--deltanet-backend", default="native")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dtype", default="fp32", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-max-windows", type=int, default=None)
    parser.add_argument("--val-max-windows", type=int, default=None)
    parser.add_argument("--test-max-windows", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prediction-split", default="val")
    parser.add_argument("--flash-fft-size", type=int, default=None)
    parser.add_argument("--downsample-prob", type=float, default=0.0)
    parser.add_argument("--downsample-min-stride", type=int, default=2)
    parser.add_argument("--downsample-max-stride", type=int, default=4)
    parser.add_argument("--amplitude-mod-prob", type=float, default=0.0)
    parser.add_argument("--flip-x-prob", type=float, default=0.0)
    parser.add_argument("--flip-y-prob", type=float, default=0.0)
    parser.add_argument("--censor-prob", type=float, default=0.0)
    parser.add_argument("--mixup-prob", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = _build_arg_parser().parse_args()
    config = TrainingConfig(
        task_name=args.task,
        activity_subset=args.activity_subset,
        context_len=args.context_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        long_conv_backend=args.long_conv_backend,
        deltanet_backend=args.deltanet_backend,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        amp=args.amp,
        dtype=args.dtype,
        num_workers=args.num_workers,
        train_max_windows=args.train_max_windows,
        val_max_windows=args.val_max_windows,
        test_max_windows=args.test_max_windows,
        device=args.device,
        seed=args.seed,
        eval_test=args.eval_test,
        data_path=args.data_path,
        output_dir=args.output_dir,
        prediction_split=args.prediction_split,
        flash_fft_size=args.flash_fft_size,
        downsample_prob=args.downsample_prob,
        downsample_min_stride=args.downsample_min_stride,
        downsample_max_stride=args.downsample_max_stride,
        amplitude_mod_prob=args.amplitude_mod_prob,
        flip_x_prob=args.flip_x_prob,
        flip_y_prob=args.flip_y_prob,
        censor_prob=args.censor_prob,
        mixup_prob=args.mixup_prob,
        mixup_alpha=args.mixup_alpha,
    )
    run_training(config)


if __name__ == "__main__":
    main()
