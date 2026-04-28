"""Configuration and task presets for Reverso models."""

from __future__ import annotations

from dataclasses import dataclass

ALLOWED_LONG_CONV_BACKENDS = frozenset({"native", "flashfftconv"})
ALLOWED_DELTANET_BACKENDS = frozenset({"native", "fla"})

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


def _normalize_channels(channels: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(channels)
    if not normalized:
        raise ValueError("Channel lists must be non-empty.")
    unknown = [channel for channel in normalized if channel not in CHANNEL_MAP]
    if unknown:
        raise ValueError(f"Unknown channel names: {unknown!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class ReversoConfig:
    """Validated configuration for a multivariate Reverso model."""

    context_len: int
    pred_len: int
    input_channels: tuple[str, ...] | list[str]
    target_channels: tuple[str, ...] | list[str]
    d_model: int
    n_layers: int
    n_heads: int = 4
    mlp_ratio: int = 4
    short_conv_kernel: int = 3
    long_conv_kernel: int | None = None
    long_conv_backend: str = "native"
    deltanet_backend: str = "native"
    flash_fft_size: int | None = None
    use_positional_embedding: bool = False
    task_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_channels", _normalize_channels(self.input_channels))
        object.__setattr__(self, "target_channels", _normalize_channels(self.target_channels))

        if self.context_len <= 0:
            raise ValueError("context_len must be positive.")
        if self.pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
        if self.short_conv_kernel <= 0:
            raise ValueError("short_conv_kernel must be positive.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.long_conv_backend not in ALLOWED_LONG_CONV_BACKENDS:
            expected = ", ".join(sorted(ALLOWED_LONG_CONV_BACKENDS))
            raise ValueError(
                f"Unknown long_conv_backend {self.long_conv_backend!r}. Expected one of: {expected}."
            )
        if self.deltanet_backend not in ALLOWED_DELTANET_BACKENDS:
            expected = ", ".join(sorted(ALLOWED_DELTANET_BACKENDS))
            raise ValueError(
                f"Unknown deltanet_backend {self.deltanet_backend!r}. Expected one of: {expected}."
            )

        long_conv_kernel = self.context_len if self.long_conv_kernel is None else self.long_conv_kernel
        if long_conv_kernel <= 0:
            raise ValueError("long_conv_kernel must be positive.")
        object.__setattr__(self, "long_conv_kernel", long_conv_kernel)
        if self.flash_fft_size is not None and self.flash_fft_size <= 0:
            raise ValueError("flash_fft_size must be positive when provided.")

        if self.task_name is not None:
            if self.task_name not in TASKS:
                raise ValueError(f"Unknown task preset: {self.task_name!r}")
            task = TASKS[self.task_name]
            if tuple(task["input_channels"]) != self.input_channels:
                raise ValueError("task_name does not match input_channels.")
            if tuple(task["target_channels"]) != self.target_channels:
                raise ValueError("task_name does not match target_channels.")

    @property
    def input_size(self) -> int:
        return len(self.input_channels)

    @property
    def output_size(self) -> int:
        return len(self.target_channels)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def resolved_flash_fft_size(self) -> int:
        if self.flash_fft_size is not None:
            return self.flash_fft_size

        fft_size = 1
        target = self.context_len + self.long_conv_kernel - 1
        while fft_size < target:
            fft_size *= 2
        return fft_size

    @classmethod
    def from_task(
        cls,
        task_name: str,
        *,
        context_len: int,
        pred_len: int,
        d_model: int,
        n_layers: int,
        **kwargs: object,
    ) -> "ReversoConfig":
        try:
            task = TASKS[task_name]
        except KeyError as error:
            raise ValueError(f"Unknown task preset: {task_name!r}") from error
        return cls(
            context_len=context_len,
            pred_len=pred_len,
            input_channels=task["input_channels"],
            target_channels=task["target_channels"],
            d_model=d_model,
            n_layers=n_layers,
            task_name=task_name,
            **kwargs,
        )
