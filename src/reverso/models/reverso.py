"""Multivariate Reverso model implementation."""

from __future__ import annotations

import inspect
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ReversoConfig

try:
    from flashfftconv import FlashFFTConv
except ImportError:
    FlashFFTConv = None

try:
    from fla.layers import DeltaNet as FLADeltaNet
except ImportError:
    FLADeltaNet = None


def validate_backend_requirements(
    config: ReversoConfig,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> None:
    """Fail fast when a requested backend cannot run in the current environment."""
    if config.long_conv_backend == "flashfftconv":
        if FlashFFTConv is None:
            raise RuntimeError(
                "long_conv_backend='flashfftconv' requires the optional 'flashfftconv' package."
            )
        if device.type != "cuda":
            raise RuntimeError("FlashFFTConv requires device='cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, so FlashFFTConv cannot run.")
        if not amp_enabled:
            raise RuntimeError(
                "FlashFFTConv requires mixed precision. Re-run with --amp --dtype bf16 or --amp --dtype fp16."
            )
        if amp_dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError("FlashFFTConv requires AMP dtype torch.float16 or torch.bfloat16.")

    if config.deltanet_backend == "fla":
        if FLADeltaNet is None:
            raise RuntimeError(
                "deltanet_backend='fla' requires the optional 'flash-linear-attention' package."
            )
        if device.type != "cuda":
            raise RuntimeError("fla DeltaNet requires device='cuda'.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, so fla DeltaNet cannot run.")


def _causal_depthwise_conv1d(x: Tensor, conv: nn.Conv1d) -> Tensor:
    """Apply a depthwise convolution with left padding only."""

    kernel_size = conv.kernel_size[0]
    x_channels_first = x.transpose(1, 2)
    padded = F.pad(x_channels_first, (kernel_size - 1, 0))
    mixed = conv(padded)
    return mixed.transpose(1, 2)


class LongConvolutionMixer(nn.Module):
    """Time-major long convolution with a short-convolution gate."""

    def __init__(self, d_model: int, *, long_kernel: int, short_kernel: int) -> None:
        super().__init__()
        self.long_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=long_kernel,
            groups=d_model,
            bias=False,
        )
        self.short_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=short_kernel,
            groups=d_model,
            bias=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.residual_scale = 0.1

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = x.float()
        long_features = _causal_depthwise_conv1d(x, self.long_conv)
        short_features = _causal_depthwise_conv1d(x, self.short_conv)
        mixed = torch.tanh(F.silu(short_features * long_features))
        mixed = self.norm(mixed).to(dtype=residual.dtype)
        return residual + self.residual_scale * mixed


class FlashFFTConvMixer(nn.Module):
    """Paper-aligned long-convolution mixer using FlashFFTConv when available."""

    def __init__(
        self,
        d_model: int,
        *,
        long_kernel: int,
        short_kernel: int,
        fft_size: int,
    ) -> None:
        super().__init__()
        if FlashFFTConv is None:
            raise ImportError(
                "FlashFFTConv backend requested, but flashfftconv is not installed. "
                "Install the package in a compatible CUDA environment first."
            )
        self.d_model = d_model
        self.long_kernel = long_kernel
        self.short_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=short_kernel,
            groups=d_model,
            bias=False,
        )
        self.long_kernel_weights = nn.Parameter(torch.empty(d_model, long_kernel, dtype=torch.float32))
        nn.init.normal_(self.long_kernel_weights, mean=0.0, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.residual_scale = 0.1
        self.fft_size = fft_size
        self.flashfftconv = FlashFFTConv(fft_size)

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_cuda:
            raise RuntimeError("FlashFFTConv backend requires a CUDA device.")
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "FlashFFTConv backend expects torch.float16 or torch.bfloat16 activations."
            )

        short_features = _causal_depthwise_conv1d(x.float(), self.short_conv)
        long_input = x.transpose(1, 2).contiguous()
        long_kernel_weights = self.long_kernel_weights.to(dtype=x.dtype)
        long_features = self.flashfftconv(long_input, long_kernel_weights).transpose(1, 2)
        mixed = torch.tanh(F.silu(short_features * long_features))
        mixed = self.norm(mixed).to(dtype=x.dtype)
        return x + self.residual_scale * mixed


class ChannelMixer(nn.Module):
    """Transformer-style channel MLP with ReLU activation."""

    def __init__(self, d_model: int, *, mlp_ratio: int) -> None:
        super().__init__()
        hidden_dim = d_model * mlp_ratio
        self.up = nn.Linear(d_model, hidden_dim)
        self.down = nn.Linear(hidden_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.residual_scale = 0.1

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = x.float()
        mixed = torch.tanh(self.down(F.relu(self.up(x))))
        mixed = self.norm(mixed).to(dtype=residual.dtype)
        return residual + self.residual_scale * mixed


class DeltaNetMixer(nn.Module):
    """DeltaNet-style linear recurrent sequence mixer."""

    def __init__(self, d_model: int, *, n_heads: int, short_kernel: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, n_heads)

        self.q_conv = nn.Conv1d(d_model, d_model, kernel_size=short_kernel, groups=d_model, bias=False)
        self.k_conv = nn.Conv1d(d_model, d_model, kernel_size=short_kernel, groups=d_model, bias=False)
        self.v_conv = nn.Conv1d(d_model, d_model, kernel_size=short_kernel, groups=d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.residual_scale = 0.1

    def _project(self, x: Tensor, proj: nn.Linear, conv: nn.Conv1d) -> Tensor:
        projected = proj(x)
        mixed = _causal_depthwise_conv1d(projected, conv)
        return torch.tanh(mixed).reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        residual = x
        weaved = x.float().clone()
        weaved[:, 0, :] = weaved[:, 0, :] + x[:, -1, :]

        q = self._project(weaved, self.q_proj, self.q_conv)
        k = self._project(weaved, self.k_proj, self.k_conv)
        v = self._project(weaved, self.v_proj, self.v_conv)
        beta = torch.sigmoid(self.beta_proj(weaved))

        q = q / math.sqrt(self.head_dim)

        state = x.new_zeros(batch_size, self.n_heads, self.head_dim, self.head_dim, dtype=torch.float32)
        outputs: list[Tensor] = []

        for step in range(seq_len):
            q_step = q[:, step, :, :]
            k_step = k[:, step, :, :]
            v_step = v[:, step, :, :]
            beta_step = beta[:, step, :].unsqueeze(-1).unsqueeze(-1)

            state_k = torch.matmul(state, k_step.unsqueeze(-1)).squeeze(-1)
            forget = state_k.unsqueeze(-1) * k_step.unsqueeze(-2)
            write = v_step.unsqueeze(-1) * k_step.unsqueeze(-2)
            state = state - beta_step * forget + beta_step * write
            state = torch.clamp(state, min=-5.0, max=5.0)

            output_step = torch.matmul(state, q_step.unsqueeze(-1)).squeeze(-1)
            outputs.append(output_step)

        mixed = torch.stack(outputs, dim=1).reshape(batch_size, seq_len, self.d_model)
        mixed = self.norm(mixed).to(dtype=residual.dtype)
        return residual + self.residual_scale * mixed


class FLADeltaNetMixer(nn.Module):
    """Wrapper around fla.layers.DeltaNet with Reverso-style state weaving."""

    def __init__(self, d_model: int, *, n_heads: int, short_kernel: int) -> None:
        super().__init__()
        if FLADeltaNet is None:
            raise ImportError(
                "fla backend requested, but flash-linear-attention is not installed."
            )

        candidate_kwargs = {
            "hidden_size": d_model,
            "num_heads": n_heads,
            "use_short_conv": True,
            "conv_size": short_kernel,
        }
        init_signature = inspect.signature(FLADeltaNet.__init__)
        filtered_kwargs = {
            key: value for key, value in candidate_kwargs.items() if key in init_signature.parameters
        }
        self.layer = FLADeltaNet(**filtered_kwargs)
        self.norm = nn.LayerNorm(d_model)
        self.residual_scale = 0.1

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_cuda:
            raise RuntimeError("fla DeltaNet backend is intended for GPU execution.")

        residual = x
        weaved = x.float().clone()
        weaved[:, 0, :] = weaved[:, 0, :] + x[:, -1, :]
        mixed = self.layer(weaved)
        if isinstance(mixed, tuple):
            mixed = mixed[0]
        mixed = self.norm(mixed).to(dtype=residual.dtype)
        return residual + self.residual_scale * mixed


class ReversoBlock(nn.Module):
    """One sequence-mixing block followed by channel mixing."""

    def __init__(self, sequence_mixer: nn.Module, channel_mixer: nn.Module) -> None:
        super().__init__()
        self.sequence_mixer = sequence_mixer
        self.channel_mixer = channel_mixer

    def forward(self, x: Tensor) -> Tensor:
        x = self.sequence_mixer(x)
        return self.channel_mixer(x)


class AttentionDecoderHead(nn.Module):
    """Attention-based decoder that maps contextualized states to P steps."""

    def __init__(self, *, context_len: int, pred_len: int, d_model: int, output_size: int) -> None:
        super().__init__()
        self.length_projection = nn.Linear(context_len, pred_len, bias=False)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, output_size)
        self.residual_scale = 0.1

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = x.float()
        query_seed = self.length_projection(x.transpose(1, 2)).transpose(1, 2)
        query = torch.tanh(self.q_proj(query_seed))
        key = torch.tanh(self.k_proj(x))
        value = torch.tanh(self.v_proj(x))

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(x.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        return (self.residual_scale * self.output_proj(attended)).to(dtype=residual.dtype)


class Reverso(nn.Module):
    """Configurable multivariate Reverso forecaster."""

    def __init__(self, config: ReversoConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_size, config.d_model)

        if config.use_positional_embedding:
            self.position_embedding = nn.Parameter(torch.zeros(1, config.context_len, config.d_model))
        else:
            self.register_parameter("position_embedding", None)

        blocks: list[nn.Module] = []
        for layer_index in range(config.n_layers):
            if layer_index % 2 == 0:
                if config.long_conv_backend == "flashfftconv":
                    sequence_mixer = FlashFFTConvMixer(
                        config.d_model,
                        long_kernel=config.long_conv_kernel,
                        short_kernel=config.short_conv_kernel,
                        fft_size=config.resolved_flash_fft_size,
                    )
                else:
                    sequence_mixer = LongConvolutionMixer(
                        config.d_model,
                        long_kernel=config.long_conv_kernel,
                        short_kernel=config.short_conv_kernel,
                    )
            else:
                if config.deltanet_backend == "fla":
                    sequence_mixer = FLADeltaNetMixer(
                        config.d_model,
                        n_heads=config.n_heads,
                        short_kernel=config.short_conv_kernel,
                    )
                else:
                    sequence_mixer = DeltaNetMixer(
                        config.d_model,
                        n_heads=config.n_heads,
                        short_kernel=config.short_conv_kernel,
                    )
            blocks.append(
                ReversoBlock(
                    sequence_mixer=sequence_mixer,
                    channel_mixer=ChannelMixer(config.d_model, mlp_ratio=config.mlp_ratio),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.decoder = AttentionDecoderHead(
            context_len=config.context_len,
            pred_len=config.pred_len,
            d_model=config.d_model,
            output_size=config.output_size,
        )

    def forward(self, context: Tensor) -> Tensor:
        if context.ndim != 3:
            raise ValueError(
                f"Expected context with shape (B, L, C_in), got tensor with shape {tuple(context.shape)!r}."
            )
        if context.shape[1] != self.config.context_len:
            raise ValueError(
                f"Expected context length {self.config.context_len}, got {context.shape[1]}."
            )
        if context.shape[2] != self.config.input_size:
            raise ValueError(
                f"Expected {self.config.input_size} input channels, got {context.shape[2]}."
            )

        x = self.input_projection(context)
        if self.position_embedding is not None:
            x = x + self.position_embedding

        for block in self.blocks:
            x = block(x)

        prediction = self.decoder(x)
        expected_shape = (context.shape[0], self.config.pred_len, self.config.output_size)
        if prediction.shape != expected_shape:
            raise RuntimeError(
                f"Decoder returned shape {tuple(prediction.shape)!r}, expected {expected_shape!r}."
            )
        return prediction


def build_model(config: ReversoConfig) -> Reverso:
    """Construct a Reverso model from a validated config."""

    return Reverso(config)
