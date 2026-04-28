"""Training and evaluation routines."""

from .config import TrainingConfig
from .metrics import ForecastMetricAccumulator, LABEL_NAMES

__all__ = [
    "ForecastMetricAccumulator",
    "LABEL_NAMES",
    "TrainingConfig",
    "build_dataloaders",
    "evaluate",
    "plot_prediction_example",
    "plot_training_curves",
    "run_training",
    "save_run_artifacts",
    "set_seed",
    "train_one_epoch",
]


def __getattr__(name: str):
    if name in {
        "build_dataloaders",
        "evaluate",
        "run_training",
        "set_seed",
        "train_one_epoch",
    }:
        from . import train

        return getattr(train, name)
    if name in {"plot_prediction_example", "plot_training_curves", "save_run_artifacts"}:
        from . import visualization

        return getattr(visualization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
