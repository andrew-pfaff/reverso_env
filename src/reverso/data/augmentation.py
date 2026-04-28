"""Paper-inspired data augmentation for Reverso forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(slots=True)
class ForecastAugmentationConfig:
    """Configurable augmentation pipeline for forecasting samples."""

    downsample_prob: float = 0.0
    downsample_min_stride: int = 2
    downsample_max_stride: int = 4
    amplitude_mod_prob: float = 0.0
    flip_x_prob: float = 0.0
    flip_y_prob: float = 0.0
    censor_prob: float = 0.0
    mixup_prob: float = 0.0
    mixup_alpha: float = 0.0

    def enabled(self) -> bool:
        return any(
            probability > 0.0
            for probability in (
                self.downsample_prob,
                self.amplitude_mod_prob,
                self.flip_x_prob,
                self.flip_y_prob,
                self.censor_prob,
                self.mixup_prob,
            )
        )

    def validate(self) -> None:
        for name in (
            "downsample_prob",
            "amplitude_mod_prob",
            "flip_x_prob",
            "flip_y_prob",
            "censor_prob",
            "mixup_prob",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if self.downsample_min_stride < 1 or self.downsample_max_stride < self.downsample_min_stride:
            raise ValueError("Invalid downsample stride range.")
        if self.mixup_alpha < 0.0:
            raise ValueError("mixup_alpha must be non-negative.")


def _resample_to_length(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly interpolate a multichannel sequence back to a target length."""
    if sequence.shape[0] == target_length:
        return sequence

    source_time = np.linspace(0.0, 1.0, num=sequence.shape[0], endpoint=True)
    target_time = np.linspace(0.0, 1.0, num=target_length, endpoint=True)
    resampled = np.empty((target_length, sequence.shape[1]), dtype=np.float32)
    for channel_index in range(sequence.shape[1]):
        resampled[:, channel_index] = np.interp(target_time, source_time, sequence[:, channel_index])
    return resampled


def augment_forecast_crop(
    crop: np.ndarray,
    *,
    context_len: int,
    config: ForecastAugmentationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply paper-inspired augmentations to a full crop, then split."""
    config.validate()
    augmented = np.asarray(crop, dtype=np.float32).copy()
    total_length = augmented.shape[0]

    if config.downsample_prob > 0.0 and torch.rand(()) < config.downsample_prob:
        stride = int(
            torch.randint(
                config.downsample_min_stride,
                config.downsample_max_stride + 1,
                (),
                dtype=torch.int64,
            ).item()
        )
        downsampled = augmented[::stride]
        if downsampled.shape[0] >= 2:
            augmented = _resample_to_length(downsampled, total_length)

    if config.amplitude_mod_prob > 0.0 and torch.rand(()) < config.amplitude_mod_prob:
        changepoint = int(torch.randint(1, total_length - 1, (), dtype=torch.int64).item())
        y1, y2, y3 = torch.normal(mean=1.0, std=0.5, size=(3,), dtype=torch.float32).tolist()
        envelope = np.interp(
            np.arange(total_length, dtype=np.float32),
            np.array([0.0, float(changepoint), float(total_length - 1)], dtype=np.float32),
            np.array([y1, y2, y3], dtype=np.float32),
        ).reshape(total_length, 1)
        augmented = augmented * envelope

    if config.flip_y_prob > 0.0 and torch.rand(()) < config.flip_y_prob:
        augmented = -augmented

    if config.flip_x_prob > 0.0 and torch.rand(()) < config.flip_x_prob:
        augmented = augmented[::-1].copy()

    context = augmented[:context_len].copy()
    target = augmented[context_len:].copy()

    if config.censor_prob > 0.0 and torch.rand(()) < config.censor_prob:
        quantile = float(torch.rand(()).item())
        threshold = np.quantile(context, quantile, axis=0, keepdims=True)
        direction = int(torch.randint(0, 3, (), dtype=torch.int64).item())
        if direction == 0:
            context = np.minimum(context, threshold)
        elif direction == 1:
            context = np.maximum(context, threshold)

    return context.astype(np.float32, copy=False), target.astype(np.float32, copy=False)


def apply_batch_mixup(
    context: Tensor,
    target: Tensor,
    labels: Tensor,
    *,
    config: ForecastAugmentationConfig,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Apply batchwise mixup to context and target tensors."""
    config.validate()
    if (
        config.mixup_prob <= 0.0
        or config.mixup_alpha <= 0.0
        or context.shape[0] < 2
        or torch.rand(()) >= config.mixup_prob
    ):
        return context, target, labels

    permutation = torch.randperm(context.shape[0], device=context.device)
    beta = torch.distributions.Beta(config.mixup_alpha, config.mixup_alpha)
    lam = beta.sample((context.shape[0],)).to(device=context.device, dtype=context.dtype)
    lam_view = lam.view(-1, 1, 1)

    mixed_context = lam_view * context + (1.0 - lam_view) * context[permutation]
    mixed_target = lam_view * target + (1.0 - lam_view) * target[permutation]
    return mixed_context, mixed_target, None
