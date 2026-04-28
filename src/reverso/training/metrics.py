"""Forecasting metrics for Reverso training and evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch
from torch import Tensor

LABEL_NAMES = {
    1: "walking",
    3: "standing",
    4: "sitting",
    5: "lying",
    8: "stairs_up",
    9: "stairs_down",
}


class ForecastMetricAccumulator:
    """Accumulates MAE metrics overall, by channel, and by activity."""

    def __init__(self, target_channels: Sequence[str]) -> None:
        self.target_channels = tuple(target_channels)
        if not self.target_channels:
            raise ValueError("target_channels must be non-empty.")

        self.total_abs_error = 0.0
        self.total_count = 0
        self.channel_abs_error = torch.zeros(len(self.target_channels), dtype=torch.float64)
        self.channel_count = 0
        self.label_abs_error: dict[int, float] = defaultdict(float)
        self.label_count: dict[int, int] = defaultdict(int)

    def update(self, prediction: Tensor, target: Tensor, labels: Tensor | None = None) -> None:
        """Update metrics from a batch of predictions and targets."""
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match, got {tuple(prediction.shape)!r} "
                f"and {tuple(target.shape)!r}."
            )
        if prediction.ndim != 3:
            raise ValueError(
                f"Expected prediction and target with shape (B, P, C), got {tuple(prediction.shape)!r}."
            )

        errors = (prediction.detach() - target.detach()).abs().to(dtype=torch.float64, device="cpu")
        self.total_abs_error += float(errors.sum().item())
        self.total_count += int(errors.numel())
        self.channel_abs_error += errors.sum(dim=(0, 1))
        self.channel_count += int(errors.shape[0] * errors.shape[1])

        if labels is None:
            return

        labels_cpu = labels.detach().to(device="cpu", dtype=torch.int64)
        if labels_cpu.ndim != 1 or labels_cpu.shape[0] != errors.shape[0]:
            raise ValueError(
                f"labels must have shape ({errors.shape[0]},), got {tuple(labels_cpu.shape)!r}."
            )

        for label in torch.unique(labels_cpu, sorted=True):
            label_id = int(label.item())
            mask = labels_cpu == label
            label_errors = errors[mask]
            self.label_abs_error[label_id] += float(label_errors.sum().item())
            self.label_count[label_id] += int(label_errors.numel())

    def compute(self) -> dict[str, float]:
        """Compute scalar MAE summaries."""
        if self.total_count == 0:
            raise ValueError("No examples were accumulated.")

        metrics = {
            "mae": self.total_abs_error / self.total_count,
            "num_forecast_points": float(self.total_count),
        }

        for channel_index, channel_name in enumerate(self.target_channels):
            metrics[f"mae_channel/{channel_name}"] = (
                self.channel_abs_error[channel_index].item() / self.channel_count
            )

        accel_indices = [
            index for index, channel_name in enumerate(self.target_channels) if channel_name.startswith("acc_")
        ]
        gyro_indices = [
            index for index, channel_name in enumerate(self.target_channels) if channel_name.startswith("gyr_")
        ]

        if accel_indices:
            accel_abs_error = self.channel_abs_error[accel_indices].sum().item()
            metrics["mae_modality/accelerometer"] = accel_abs_error / (
                self.channel_count * len(accel_indices)
            )
        if gyro_indices:
            gyro_abs_error = self.channel_abs_error[gyro_indices].sum().item()
            metrics["mae_modality/gyroscope"] = gyro_abs_error / (
                self.channel_count * len(gyro_indices)
            )

        for label_id, abs_error in sorted(self.label_abs_error.items()):
            label_name = LABEL_NAMES.get(label_id, f"label_{label_id}")
            metrics[f"mae_activity/{label_name}"] = abs_error / self.label_count[label_id]

        return metrics
