"""Visualization and artifact-saving helpers for Reverso training runs."""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/reverso_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .metrics import LABEL_NAMES


def _json_ready(value: Any) -> Any:
    """Convert config values to JSON-serializable equivalents."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def save_run_metadata(
    output_dir: Path,
    *,
    config: TrainingConfig,
    history: list[dict[str, float]],
    test_metrics: dict[str, float] | None,
) -> dict[str, Path]:
    """Save config and metric history to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    history_path = output_dir / "history.json"
    summary_path = output_dir / "summary.json"

    config_payload = _json_ready(asdict(config))
    history_payload = _json_ready(history)
    summary_payload = {
        "final_epoch": history[-1] if history else None,
        "test_metrics": _json_ready(test_metrics),
    }

    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n")
    history_path.write_text(json.dumps(history_payload, indent=2, sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")

    return {
        "config": config_path,
        "history": history_path,
        "summary": summary_path,
    }


def _metric_series(history: list[dict[str, float]], key: str) -> list[float]:
    """Extract one metric series across epochs."""
    return [float(epoch_metrics[key]) for epoch_metrics in history if key in epoch_metrics]


def plot_training_curves(history: list[dict[str, float]], output_dir: Path) -> Path:
    """Plot loss and MAE curves over epochs."""
    if not history:
        raise ValueError("history must contain at least one epoch.")

    epochs = [int(epoch_metrics["epoch"]) for epoch_metrics in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    for axis, metric_name, title in [
        (axes[0], "loss", "Loss"),
        (axes[1], "mae", "Overall MAE"),
        (axes[2], "mae_modality/accelerometer", "Accelerometer MAE"),
        (axes[3], "mae_modality/gyroscope", "Gyroscope MAE"),
    ]:
        train_key = f"train_{metric_name}"
        val_key = f"val_{metric_name}"
        plotted = False
        if train_key in history[0]:
            axis.plot(epochs, _metric_series(history, train_key), marker="o", label="train")
            plotted = True
        if val_key in history[0]:
            axis.plot(epochs, _metric_series(history, val_key), marker="o", label="val")
            plotted = True
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric_name.split("/")[-1])
        axis.grid(alpha=0.3)
        if plotted:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "N/A for this task", ha="center", va="center", transform=axis.transAxes)

    fig.suptitle("Reverso Training Curves", fontsize=15)
    fig.tight_layout()

    output_path = output_dir / "training_curves.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_final_channel_metrics(history: list[dict[str, float]], output_dir: Path) -> Path:
    """Plot grouped train/val MAE bars for each target channel."""
    if not history:
        raise ValueError("history must contain at least one epoch.")
    final_metrics = history[-1]

    channel_names = sorted(
        key.removeprefix("val_mae_channel/")
        for key in final_metrics
        if key.startswith("val_mae_channel/")
    )
    if not channel_names:
        raise ValueError("No channel metrics were found in history.")

    train_values = [float(final_metrics[f"train_mae_channel/{channel_name}"]) for channel_name in channel_names]
    val_values = [float(final_metrics[f"val_mae_channel/{channel_name}"]) for channel_name in channel_names]

    x = np.arange(len(channel_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width / 2, train_values, width=width, label="train", color="#457b9d")
    ax.bar(x + width / 2, val_values, width=width, label="val", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=20)
    ax.set_ylabel("MAE")
    ax.set_title("Final Epoch Channel MAE")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / "channel_mae.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_final_activity_metrics(history: list[dict[str, float]], output_dir: Path) -> Path:
    """Plot grouped train/val MAE bars for each activity label present."""
    if not history:
        raise ValueError("history must contain at least one epoch.")
    final_metrics = history[-1]

    activity_names = sorted(
        {
            key.removeprefix("train_mae_activity/")
            for key in final_metrics
            if key.startswith("train_mae_activity/")
        }
        | {
            key.removeprefix("val_mae_activity/")
            for key in final_metrics
            if key.startswith("val_mae_activity/")
        }
    )
    if not activity_names:
        raise ValueError("No activity metrics were found in history.")

    train_values = [float(final_metrics.get(f"train_mae_activity/{name}", np.nan)) for name in activity_names]
    val_values = [float(final_metrics.get(f"val_mae_activity/{name}", np.nan)) for name in activity_names]

    x = np.arange(len(activity_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width / 2, train_values, width=width, label="train", color="#2a9d8f")
    ax.bar(x + width / 2, val_values, width=width, label="val", color="#e9c46a")
    ax.set_xticks(x)
    ax.set_xticklabels(activity_names, rotation=20)
    ax.set_ylabel("MAE")
    ax.set_title("Final Epoch Activity MAE")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / "activity_mae.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_prediction_example(
    model: nn.Module,
    dataloader: DataLoader[Any],
    *,
    device: torch.device,
    input_channels: Sequence[str],
    target_channels: Sequence[str],
    split_name: str,
    output_dir: Path,
    context_tail_points: int = 128,
    target_label_name: str | None = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> Path:
    """Plot one forecast against the target sequence for each target channel."""
    target_label_id = None
    if target_label_name is not None:
        reverse_label_names = {name: label_id for label_id, name in LABEL_NAMES.items()}
        if target_label_name not in reverse_label_names:
            expected = ", ".join(sorted(reverse_label_names))
            raise ValueError(
                f"Unknown target_label_name {target_label_name!r}. Expected one of: {expected}."
            )
        target_label_id = reverse_label_names[target_label_name]

    selected_batch = None
    selected_index = None
    for batch in dataloader:
        labels = batch["label"]
        if target_label_id is None:
            selected_batch = batch
            selected_index = 0
            break
        matches = torch.nonzero(labels == target_label_id, as_tuple=False)
        if matches.numel() > 0:
            selected_batch = batch
            selected_index = int(matches[0].item())
            break

    if selected_batch is None or selected_index is None:
        label_text = target_label_name if target_label_name is not None else "any label"
        raise ValueError(f"Could not find a prediction example for {label_text!r} in {split_name!r}.")

    context = selected_batch["context"].to(device=device, non_blocking=True)
    target = selected_batch["target"]
    labels = selected_batch["label"]

    model.eval()
    with torch.no_grad():
        if amp_enabled:
            autocast_context = torch.autocast(device_type=device.type, dtype=amp_dtype)
        else:
            autocast_context = nullcontext()
        with autocast_context:
            prediction = model(context).detach().to(dtype=torch.float32).cpu()

    context_cpu = context.cpu()
    target_cpu = target.cpu()
    label_id = int(labels[selected_index].item())
    label_name = LABEL_NAMES.get(label_id, f"label_{label_id}")

    context_example = context_cpu[selected_index]
    target_example = target_cpu[selected_index]
    prediction_example = prediction[selected_index]

    context_tail = min(context_tail_points, context_example.shape[0])
    input_channel_to_index = {channel_name: index for index, channel_name in enumerate(input_channels)}

    num_channels = len(target_channels)
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2.8 * num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]

    context_time = np.arange(-context_tail, 0)
    future_time = np.arange(target_example.shape[0])

    for axis, channel_index, channel_name in zip(axes, range(num_channels), target_channels):
        if channel_name in input_channel_to_index:
            context_index = input_channel_to_index[channel_name]
            axis.plot(
                context_time,
                context_example[-context_tail:, context_index].to(dtype=torch.float32).numpy(),
                label="context",
                color="#6c757d",
            )
        axis.plot(
            future_time,
            target_example[:, channel_index].to(dtype=torch.float32).numpy(),
            label="truth",
            color="#1d3557",
        )
        axis.plot(
            future_time,
            prediction_example[:, channel_index].to(dtype=torch.float32).numpy(),
            label="prediction",
            color="#e63946",
            linestyle="--",
        )
        axis.axvline(0, color="black", linewidth=1, alpha=0.4)
        axis.set_ylabel(channel_name)
        axis.grid(alpha=0.3)

    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Time step relative to forecast start")
    fig.suptitle(f"{split_name.title()} Prediction Example ({label_name})", fontsize=15)
    fig.tight_layout()

    suffix = f"_{target_label_name}" if target_label_name is not None else ""
    output_path = output_dir / f"prediction_example_{split_name}{suffix}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_run_artifacts(
    output_dir: Path,
    *,
    config: TrainingConfig,
    history: list[dict[str, float]],
    test_metrics: dict[str, float] | None,
    model: nn.Module,
    dataloaders: dict[str, DataLoader[Any]],
    device: torch.device,
    input_channels: Sequence[str],
    target_channels: Sequence[str],
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, Path]:
    """Save JSON summaries and plot files for a completed training run."""
    paths = save_run_metadata(output_dir, config=config, history=history, test_metrics=test_metrics)
    paths["training_curves"] = plot_training_curves(history, output_dir)
    paths["channel_mae"] = plot_final_channel_metrics(history, output_dir)
    paths["activity_mae"] = plot_final_activity_metrics(history, output_dir)

    prediction_split = config.prediction_split
    if prediction_split not in dataloaders:
        available = ", ".join(sorted(dataloaders))
        raise ValueError(
            f"prediction_split {prediction_split!r} is not available. Have: {available}."
        )
    prediction_loader = dataloaders[prediction_split]

    paths["prediction_example"] = plot_prediction_example(
        model,
        prediction_loader,
        device=device,
        input_channels=input_channels,
        target_channels=target_channels,
        split_name=prediction_split,
        output_dir=output_dir,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    for label_name in ("stairs_up", "stairs_down"):
        try:
            paths[f"prediction_example_{label_name}"] = plot_prediction_example(
                model,
                prediction_loader,
                device=device,
                input_channels=input_channels,
                target_channels=target_channels,
                split_name=prediction_split,
                output_dir=output_dir,
                target_label_name=label_name,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
        except ValueError:
            continue
    return paths
