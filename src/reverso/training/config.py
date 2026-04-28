"""Training configuration for the UCI HAR Reverso baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from reverso.data import (
    DEFAULT_CONTEXT_LEN,
    DEFAULT_PRED_LEN,
    DEFAULT_TASK,
    ForecastAugmentationConfig,
    resolve_activity_subset,
    resolve_channel_names,
    resolve_task,
)
from reverso.models import ReversoConfig


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for model training and evaluation."""

    task_name: str = DEFAULT_TASK
    context_len: int = DEFAULT_CONTEXT_LEN
    pred_len: int = DEFAULT_PRED_LEN
    input_channels: Sequence[str] | None = None
    target_channels: Sequence[str] | None = None
    activity_subset: str = "all"
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: int = 4
    short_conv_kernel: int = 3
    long_conv_kernel: int | None = None
    long_conv_backend: str = "native"
    deltanet_backend: str = "native"
    flash_fft_size: int | None = None
    use_positional_embedding: bool = False
    batch_size: int = 32
    num_epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip_norm: float | None = 1.0
    amp: bool = False
    dtype: str = "fp32"
    num_workers: int = 0
    seed: int = 42
    train_max_windows: int | None = None
    val_max_windows: int | None = None
    test_max_windows: int | None = None
    device: str | None = None
    data_path: str | Path | None = None
    eval_test: bool = False
    output_dir: str | Path | None = None
    prediction_split: str = "val"
    downsample_prob: float = 0.0
    downsample_min_stride: int = 2
    downsample_max_stride: int = 4
    amplitude_mod_prob: float = 0.0
    flip_x_prob: float = 0.0
    flip_y_prob: float = 0.0
    censor_prob: float = 0.0
    mixup_prob: float = 0.0
    mixup_alpha: float = 0.0

    def resolved_channels(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Resolve the configured input and target channel names."""
        if self.input_channels is None and self.target_channels is None:
            task = resolve_task(self.task_name)
            return (
                tuple(task["input_channels"]),
                tuple(task["target_channels"]),
            )

        if self.input_channels is None or self.target_channels is None:
            raise ValueError(
                "input_channels and target_channels must be provided together when "
                "overriding a task preset."
            )

        return (
            resolve_channel_names(self.input_channels),
            resolve_channel_names(self.target_channels),
        )

    def resolved_activity_subset(self) -> tuple[int, ...]:
        """Resolve the configured activity subset preset."""
        return resolve_activity_subset(self.activity_subset)

    def resolved_device(self) -> torch.device:
        """Select the requested device or a sensible default."""
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def resolved_output_dir(self) -> Path | None:
        """Resolve the output directory if one was configured."""
        if self.output_dir is None:
            return None
        return Path(self.output_dir).expanduser()

    def resolved_training_dtype(self) -> torch.dtype:
        """Resolve the configured activation dtype."""
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        try:
            return dtype_map[self.dtype]
        except KeyError as error:
            expected = ", ".join(sorted(dtype_map))
            raise ValueError(f"Unknown dtype {self.dtype!r}. Expected one of: {expected}.") from error

    def resolved_amp_dtype(self) -> torch.dtype | None:
        """Resolve the autocast dtype when AMP is enabled."""
        if not self.amp:
            return None
        dtype = self.resolved_training_dtype()
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("AMP requires dtype='fp16' or dtype='bf16'.")
        return dtype

    def make_augmentation_config(self) -> ForecastAugmentationConfig | None:
        """Build the configured augmentation pipeline, if any."""
        config = ForecastAugmentationConfig(
            downsample_prob=self.downsample_prob,
            downsample_min_stride=self.downsample_min_stride,
            downsample_max_stride=self.downsample_max_stride,
            amplitude_mod_prob=self.amplitude_mod_prob,
            flip_x_prob=self.flip_x_prob,
            flip_y_prob=self.flip_y_prob,
            censor_prob=self.censor_prob,
            mixup_prob=self.mixup_prob,
            mixup_alpha=self.mixup_alpha,
        )
        config.validate()
        if not config.enabled():
            return None
        return config

    def make_model_config(self) -> ReversoConfig:
        """Build a validated model config from the training config."""
        input_channels, target_channels = self.resolved_channels()
        return ReversoConfig(
            context_len=self.context_len,
            pred_len=self.pred_len,
            input_channels=input_channels,
            target_channels=target_channels,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            mlp_ratio=self.mlp_ratio,
            short_conv_kernel=self.short_conv_kernel,
            long_conv_kernel=self.long_conv_kernel,
            long_conv_backend=self.long_conv_backend,
            deltanet_backend=self.deltanet_backend,
            flash_fft_size=self.flash_fft_size,
            use_positional_embedding=self.use_positional_embedding,
            task_name=self.task_name if (
                self.input_channels is None and self.target_channels is None
            ) else None,
        )
