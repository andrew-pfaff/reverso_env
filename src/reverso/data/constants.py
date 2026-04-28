"""Shared constants and helpers for the UCI HAR Reverso baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

CHANNEL_MAP = {
    "acc_x": 0,
    "acc_y": 1,
    "acc_z": 2,
    "gyr_x": 3,
    "gyr_y": 4,
    "gyr_z": 5,
}

TASKS = {
    "acc_to_acc": {
        "input_channels": ["acc_x", "acc_y", "acc_z"],
        "target_channels": ["acc_x", "acc_y", "acc_z"],
    },
    "gyro_to_gyro": {
        "input_channels": ["gyr_x", "gyr_y", "gyr_z"],
        "target_channels": ["gyr_x", "gyr_y", "gyr_z"],
    },
    "all_to_acc": {
        "input_channels": ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"],
        "target_channels": ["acc_x", "acc_y", "acc_z"],
    },
    "all_to_gyro": {
        "input_channels": ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"],
        "target_channels": ["gyr_x", "gyr_y", "gyr_z"],
    },
    "all_to_all": {
        "input_channels": ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"],
        "target_channels": ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"],
    },
}

LABEL_NAMES = {
    1: "walking",
    3: "standing",
    4: "sitting",
    5: "lying",
    8: "stairs_up",
    9: "stairs_down",
}

LABEL_NAME_TO_ID = {name: label_id for label_id, name in LABEL_NAMES.items()}

ACTIVITY_SUBSETS = {
    "all": tuple(sorted(LABEL_NAMES)),
    "stairs_up": (LABEL_NAME_TO_ID["stairs_up"],),
    "stairs_down": (LABEL_NAME_TO_ID["stairs_down"],),
    "stairs_both": (
        LABEL_NAME_TO_ID["stairs_up"],
        LABEL_NAME_TO_ID["stairs_down"],
    ),
}

TRAIN_SUBJECT_IDS = (
    2303,
    2305,
    2306,
    2307,
    2311,
    2314,
    2316,
    2317,
    2319,
    2321,
    2322,
    2325,
    2326,
    2327,
    2328,
    2329,
)
VAL_SUBJECT_IDS = (2301, 2308, 2315, 2323, 2330)
TEST_SUBJECT_IDS = (2302, 2304, 2309, 2310, 2312, 2313, 2318, 2320, 2324)

SPLIT_SUBJECT_IDS = {
    "train": TRAIN_SUBJECT_IDS,
    "val": VAL_SUBJECT_IDS,
    "test": TEST_SUBJECT_IDS,
}

DEFAULT_TASK = "all_to_all"
DEFAULT_CONTEXT_LEN = 384
DEFAULT_PRED_LEN = 64
WINDOW_LEN = 512
WINDOW_DATASET_KEY = "windows"
LABELS_DATASET_KEY = "labels"
SUBJECT_IDS_DATASET_KEY = "subject_ids"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "uci_har" / "windows.h5"
VALID_CROP_MODES = frozenset({"split_default", "random", "fixed"})


def normalize_split(split: str) -> str:
    """Validate and normalize the requested split name."""
    split_name = split.strip().lower()
    if split_name not in SPLIT_SUBJECT_IDS:
        expected = ", ".join(sorted(SPLIT_SUBJECT_IDS))
        raise ValueError(f"Unknown split {split!r}. Expected one of: {expected}.")
    return split_name


def get_split_subject_ids(split: str) -> tuple[int, ...]:
    """Return the fixed subject IDs for a split."""
    return SPLIT_SUBJECT_IDS[normalize_split(split)]


def default_crop_mode_for_split(split: str) -> str:
    """Return the baseline crop mode for the split."""
    split_name = normalize_split(split)
    if split_name == "train":
        return "random"
    return "fixed"


def resolve_crop_mode(split: str, crop_mode: str | None) -> str:
    """Resolve the effective crop mode, defaulting to the split policy."""
    if crop_mode is None:
        return default_crop_mode_for_split(split)

    crop_mode_name = crop_mode.strip().lower()
    if crop_mode_name == "split_default":
        return default_crop_mode_for_split(split)
    if crop_mode_name not in VALID_CROP_MODES:
        expected = ", ".join(sorted(VALID_CROP_MODES))
        raise ValueError(f"Unknown crop_mode {crop_mode!r}. Expected one of: {expected}.")
    return crop_mode_name


def validate_context_and_pred_lens(context_len: int, pred_len: int) -> None:
    """Ensure the requested crop fits inside a stored window."""
    if context_len <= 0:
        raise ValueError(f"context_len must be positive, got {context_len}.")
    if pred_len <= 0:
        raise ValueError(f"pred_len must be positive, got {pred_len}.")
    if context_len + pred_len > WINDOW_LEN:
        raise ValueError(
            "context_len + pred_len must be <= "
            f"{WINDOW_LEN}, got {context_len + pred_len}."
        )


def resolve_task(task: str) -> dict[str, list[str]]:
    """Return a copy of the configured channel lists for a task preset."""
    task_name = task.strip().lower()
    if task_name not in TASKS:
        expected = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task preset {task!r}. Expected one of: {expected}.")
    preset = TASKS[task_name]
    return {
        "input_channels": list(preset["input_channels"]),
        "target_channels": list(preset["target_channels"]),
    }


def resolve_activity_subset(activity_subset: str | None) -> tuple[int, ...]:
    """Resolve an activity subset preset to its label IDs."""
    if activity_subset is None:
        return ACTIVITY_SUBSETS["all"]

    subset_name = activity_subset.strip().lower()
    if subset_name not in ACTIVITY_SUBSETS:
        expected = ", ".join(sorted(ACTIVITY_SUBSETS))
        raise ValueError(
            f"Unknown activity_subset {activity_subset!r}. Expected one of: {expected}."
        )
    return ACTIVITY_SUBSETS[subset_name]


def resolve_channel_names(channel_names: Sequence[str]) -> tuple[str, ...]:
    """Validate a channel-name sequence while preserving order."""
    if isinstance(channel_names, str):
        raise TypeError("Channel configuration must be a sequence of channel names, not a string.")
    resolved = []
    for channel_name in channel_names:
        if channel_name not in CHANNEL_MAP:
            expected = ", ".join(CHANNEL_MAP)
            raise ValueError(
                f"Unknown channel name {channel_name!r}. Expected one of: {expected}."
            )
        resolved.append(channel_name)
    if not resolved:
        raise ValueError("At least one channel must be selected.")
    return tuple(resolved)


def resolve_channel_indices(channel_names: Sequence[str]) -> tuple[int, ...]:
    """Map channel names to their canonical indices."""
    return tuple(CHANNEL_MAP[name] for name in resolve_channel_names(channel_names))


def _validate_split_subject_ids() -> None:
    seen: set[int] = set()
    for split_name, subject_ids in SPLIT_SUBJECT_IDS.items():
        overlap = seen.intersection(subject_ids)
        if overlap:
            overlap_list = ", ".join(str(subject_id) for subject_id in sorted(overlap))
            raise ValueError(f"Split {split_name!r} overlaps existing split subjects: {overlap_list}.")
        seen.update(subject_ids)


_validate_split_subject_ids()
