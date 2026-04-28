"""Dataset implementation for the UCI HAR Reverso baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import ForecastAugmentationConfig, augment_forecast_crop
from .constants import (
    CHANNEL_MAP,
    DEFAULT_CONTEXT_LEN,
    DEFAULT_DATA_PATH,
    DEFAULT_PRED_LEN,
    DEFAULT_TASK,
    LABELS_DATASET_KEY,
    SUBJECT_IDS_DATASET_KEY,
    WINDOW_DATASET_KEY,
    WINDOW_LEN,
    get_split_subject_ids,
    normalize_split,
    resolve_activity_subset,
    resolve_channel_indices,
    resolve_channel_names,
    resolve_crop_mode,
    resolve_task,
    validate_context_and_pred_lens,
)


class UCIHARReversoDataset(Dataset[dict[str, torch.Tensor]]):
    """HDF5-backed dataset for the UCI HAR Reverso forecasting baseline."""

    def __init__(
        self,
        *,
        split: str,
        context_len: int = DEFAULT_CONTEXT_LEN,
        pred_len: int = DEFAULT_PRED_LEN,
        input_channels: Sequence[str] | None = None,
        target_channels: Sequence[str] | None = None,
        crop_mode: str | None = None,
        task: str = DEFAULT_TASK,
        data_path: str | Path | None = None,
        fixed_crop_start: int = 0,
        activity_subset: str | None = None,
        augmentation: ForecastAugmentationConfig | None = None,
    ) -> None:
        self.split = normalize_split(split)
        self.context_len = int(context_len)
        self.pred_len = int(pred_len)
        validate_context_and_pred_lens(self.context_len, self.pred_len)

        task_preset = resolve_task(task)
        self.task = task.strip().lower()
        self.input_channels = resolve_channel_names(
            input_channels if input_channels is not None else task_preset["input_channels"]
        )
        self.target_channels = resolve_channel_names(
            target_channels if target_channels is not None else task_preset["target_channels"]
        )
        self.input_channel_indices = resolve_channel_indices(self.input_channels)
        self.target_channel_indices = resolve_channel_indices(self.target_channels)

        self.crop_mode = resolve_crop_mode(self.split, crop_mode)
        self.fixed_crop_start = int(fixed_crop_start)
        self.max_crop_start = WINDOW_LEN - (self.context_len + self.pred_len)
        self.activity_subset = activity_subset.strip().lower() if activity_subset is not None else "all"
        self.allowed_label_ids = resolve_activity_subset(activity_subset)
        self.augmentation = augmentation
        if self.augmentation is not None:
            self.augmentation.validate()
        if not 0 <= self.fixed_crop_start <= self.max_crop_start:
            raise ValueError(
                f"fixed_crop_start must be in [0, {self.max_crop_start}], got {self.fixed_crop_start}."
            )

        self._h5_file: h5py.File | None = None
        self._windows: h5py.Dataset | None = None

        self.data_path = (
            Path(data_path).expanduser() if data_path is not None else DEFAULT_DATA_PATH
        )
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Could not find HDF5 dataset at {self.data_path}.")

        self.split_subject_ids = get_split_subject_ids(self.split)

        with h5py.File(self.data_path, "r") as h5_file:
            windows = h5_file[WINDOW_DATASET_KEY]
            if windows.shape[1] != WINDOW_LEN:
                raise ValueError(
                    f"Expected {WINDOW_DATASET_KEY!r} windows of length {WINDOW_LEN}, "
                    f"found shape {windows.shape}."
                )
            if windows.shape[2] != len(CHANNEL_MAP):
                raise ValueError(
                    f"Expected {WINDOW_DATASET_KEY!r} to have {len(CHANNEL_MAP)} channels, "
                    f"found {windows.shape[2]}."
                )

            labels = np.asarray(h5_file[LABELS_DATASET_KEY][:], dtype=np.int64)
            subject_ids = np.asarray(h5_file[SUBJECT_IDS_DATASET_KEY][:], dtype=np.int64)

        mask = np.isin(subject_ids, self.split_subject_ids) & np.isin(labels, self.allowed_label_ids)
        self.window_indices = np.flatnonzero(mask).astype(np.int64, copy=False)
        if self.window_indices.size == 0:
            raise ValueError(
                f"No windows found for split {self.split!r} and activity subset "
                f"{self.activity_subset!r}."
            )
        self.labels = labels[self.window_indices]
        self.subject_ids = subject_ids[self.window_indices]

    def __len__(self) -> int:
        return int(self.window_indices.shape[0])

    @property
    def n_input_channels(self) -> int:
        return len(self.input_channel_indices)

    @property
    def n_target_channels(self) -> int:
        return len(self.target_channel_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if isinstance(index, torch.Tensor):
            index = int(index.item())
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} is out of range for dataset of length {len(self)}.")

        window_index = int(self.window_indices[index])
        crop_start = self._select_crop_start()
        crop_stop = crop_start + self.context_len + self.pred_len
        window = np.asarray(self._get_windows()[window_index], dtype=np.float32)
        crop = window[crop_start:crop_stop]

        if self.augmentation is not None and self.crop_mode == "random":
            context_full, target_full = augment_forecast_crop(
                crop,
                context_len=self.context_len,
                config=self.augmentation,
            )
        else:
            context_full = crop[: self.context_len]
            target_full = crop[self.context_len :]

        context_np = np.take(context_full, self.input_channel_indices, axis=1)
        target_np = np.take(target_full, self.target_channel_indices, axis=1)

        context = (torch.from_numpy(context_np.copy()) + 1.0) / 2.0
        target = (torch.from_numpy(target_np.copy()) + 1.0) / 2.0

        return {
            "context": context,
            "target": target,
            "label": torch.tensor(int(self.labels[index]), dtype=torch.int64),
            "subject_id": torch.tensor(int(self.subject_ids[index]), dtype=torch.int64),
            "window_index": torch.tensor(window_index, dtype=torch.int64),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_windows"] = None
        return state

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self.split!r}, "
            f"num_windows={len(self)}, "
            f"context_len={self.context_len}, "
            f"pred_len={self.pred_len}, "
            f"input_channels={list(self.input_channels)!r}, "
            f"target_channels={list(self.target_channels)!r}, "
            f"crop_mode={self.crop_mode!r}, "
            f"activity_subset={self.activity_subset!r})"
        )

    def close(self) -> None:
        h5_file = getattr(self, "_h5_file", None)
        if h5_file is not None:
            h5_file.close()
        self._h5_file = None
        self._windows = None

    def _get_windows(self) -> h5py.Dataset:
        if self._windows is None:
            self._h5_file = h5py.File(self.data_path, "r")
            self._windows = self._h5_file[WINDOW_DATASET_KEY]
        return self._windows

    def _select_crop_start(self) -> int:
        if self.crop_mode == "fixed":
            return self.fixed_crop_start
        return int(torch.randint(self.max_crop_start + 1, (), dtype=torch.int64).item())

    def __del__(self) -> None:
        self.close()


ReversoDataset = UCIHARReversoDataset

__all__ = ["ReversoDataset", "UCIHARReversoDataset"]
