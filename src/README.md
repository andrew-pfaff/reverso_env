# `src/` Package Notes

This directory contains the current local Reverso implementation for the UCI HAR forecasting experiments.

## Important Caveat

The current model code is **not yet a fully faithful implementation** of the Reverso paper,
*Reverso: Efficient Time Series Foundation Models for Zero-shot Forecasting*.

What is still missing from a paper-faithful implementation:

- full runtime validation of the `FlashFFTConv` long convolution backend on a real CUDA machine
- full runtime validation of the `DeltaNet` layer imported from `fla.layers`, on a real CUDA machine

What the current implementation uses instead:

- a native PyTorch depthwise long-convolution mixer
- a native PyTorch DeltaNet-style mixer implemented in `src/reverso/models/reverso.py`

Why this version exists:

- it is easier to run locally
- it works on CPU
- it supports smoke tests and small pilot runs on UCI HAR without requiring the GPU-specific Reverso dependencies

The intended future upgrade path is:

- keep the current data and training interfaces
- swap in `FlashFFTConv` for the long-convolution block
- swap in `fla.layers.DeltaNet` for the handwritten DeltaNet block

The codebase now includes:

- backend selection flags for `flashfftconv` and `fla`
- CUDA fail-fast checks before training starts
- AMP support for `fp16` and `bf16`
- optional GPU dependencies in `pyproject.toml`

## Scope Of The Current Implementation

This codebase currently supports:

- multivariate UCI HAR forecasting with configurable channel subsets
- task presets such as `acc_to_acc`, `all_to_acc`, and `all_to_all`
- deterministic subject-wise train/val/test splits
- a Reverso-style hybrid model with alternating long-conv and DeltaNet-style blocks
- a training loop with MAE-based evaluation
- smoke tests for dataset loading, batching, model forward, and one optimizer step

The shared data/model contract is defined in:

- `docs/uci_har_reverso_requirements.md`

## Package Layout

- `reverso/data/`: UCI HAR dataset loader, channel/task presets, split definitions
- `reverso/models/`: configurable multivariate Reverso approximation
- `reverso/training/`: training config, metrics, dataloader integration, CLI

## Example: Load A Dataset

After installing the project in editable mode with `pip install -e .`:

```python
from reverso.data import UCIHARReversoDataset

dataset = UCIHARReversoDataset(
    split="train",
    task="all_to_all",
    context_len=384,
    pred_len=64,
)

sample = dataset[0]
print(sample["context"].shape)  # (384, 6)
print(sample["target"].shape)   # (64, 6)
print(sample["label"])          # scalar activity label
```

## Example: Build A Nano Model

```python
from reverso.models import ReversoConfig, build_model

config = ReversoConfig.from_task(
    "all_to_all",
    context_len=384,
    pred_len=64,
    d_model=32,
    n_layers=2,
)

model = build_model(config)
print(model.config)
```

## Example: Run A Forward Pass

```python
import torch

from reverso.data import UCIHARReversoDataset
from reverso.models import ReversoConfig, build_model

dataset = UCIHARReversoDataset(split="train", task="all_to_acc")
sample = dataset[0]

config = ReversoConfig.from_task(
    "all_to_acc",
    context_len=384,
    pred_len=64,
    d_model=32,
    n_layers=2,
)
model = build_model(config)

context = sample["context"].unsqueeze(0)  # (1, 384, 6)
prediction = model(context)
print(prediction.shape)  # (1, 64, 3)
```

## Example: Run A Small Nano Pilot

```bash
.venv/bin/python -m reverso.training.train \
  --task all_to_all \
  --d-model 32 \
  --n-layers 2 \
  --batch-size 16 \
  --epochs 5 \
  --train-max-windows 256 \
  --val-max-windows 128 \
  --device cpu \
  --output-dir runs/nano_all_to_all
```

This is the recommended first pilot because it:

- uses the existing smoke-tested path
- keeps the model small
- keeps the dataset subset small enough for local debugging

## Example: Run A Stair-Only Pilot

```bash
.venv/bin/python -m reverso.training.train \
  --task all_to_acc \
  --activity-subset stairs_both \
  --d-model 32 \
  --n-layers 2 \
  --batch-size 16 \
  --epochs 5 \
  --train-max-windows 256 \
  --device cpu \
  --output-dir runs/stairs_only_all_to_acc
```

Useful activity subset presets:

- `all`
- `stairs_up`
- `stairs_down`
- `stairs_both`

## Example: Run A Stair-Only Pilot With Augmentation

```bash
.venv/bin/python -m reverso.training.train \
  --task all_to_acc \
  --activity-subset stairs_both \
  --d-model 32 \
  --n-layers 2 \
  --batch-size 16 \
  --epochs 5 \
  --train-max-windows 256 \
  --device cpu \
  --downsample-prob 0.3 \
  --amplitude-mod-prob 0.3 \
  --censor-prob 0.3 \
  --mixup-prob 0.3 \
  --mixup-alpha 0.2 \
  --output-dir runs/stairs_augmented_all_to_acc
```

## Example: Prepare A CUDA Run

Install the optional GPU dependencies:

```bash
.venv/bin/pip install -e '.[gpu]'
.venv/bin/pip install --no-build-isolation \
  "git+https://github.com/HazyResearch/flash-fft-conv.git#subdirectory=csrc/flashfftconv"
```

This pulls `flash-linear-attention` from PyPI and installs `flashfftconv` from the upstream GitHub source.
The separate `--no-build-isolation` step is required because the FlashFFTConv build needs to see the active `torch` install.

Then launch the paper-aligned backend path with mixed precision:

```bash
.venv/bin/python -m reverso.training.train \
  --task all_to_acc \
  --activity-subset stairs_both \
  --d-model 32 \
  --n-layers 2 \
  --batch-size 16 \
  --epochs 5 \
  --train-max-windows 256 \
  --val-max-windows 128 \
  --device cuda \
  --amp \
  --dtype bf16 \
  --long-conv-backend flashfftconv \
  --deltanet-backend fla \
  --flash-fft-size 1024 \
  --output-dir runs/stairs_gpu_all_to_acc
```

Notes:

- `FlashFFTConv` requires CUDA plus `--amp --dtype bf16` or `--amp --dtype fp16`
- the training entrypoint now fails fast if you request `flashfftconv` without CUDA or AMP
- `bf16` is the safer first choice when the GPU supports it

The current augmentation path is inspired by the paper and supports:

- downsampling
- amplitude modulation
- flip-x (temporal reversal)
- flip-y (sign inversion)
- censor
- batch mixup

For IMU data, conservative defaults are recommended. In particular, `flip-x` and `flip-y` should be treated as optional because they are more natural for generic scalar time series than for gravity-retaining inertial signals.

When `--output-dir` is provided, the training run writes:

- `config.json`
- `history.json`
- `summary.json`
- `checkpoints/best.pt`
- `checkpoints/final.pt`
- `training_curves.png`
- `channel_mae.png`
- `activity_mae.png`
- `prediction_example_<split>.png`
- `prediction_example_<split>_stairs_up.png`
- `prediction_example_<split>_stairs_down.png`

## Example: Compare Tasks

```bash
.venv/bin/python -m reverso.training.train \
  --task acc_to_acc \
  --d-model 32 \
  --n-layers 2 \
  --batch-size 16 \
  --epochs 5 \
  --train-max-windows 256 \
  --val-max-windows 128 \
  --device cpu
```

Other supported task presets:

- `gyro_to_gyro`
- `all_to_acc`
- `all_to_gyro`
- `all_to_all`

## Example: Run Smoke Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

These tests currently verify:

- dataset loading and batching
- model forward shape
- metric accumulation
- one optimizer step plus validation pass

## When To Revisit The Paper-Faithful Rewrite

The right time to replace the current approximation with the paper-aligned backend is after:

- the UCI HAR data pipeline is stable
- the training loop is producing sensible pilot results
- you have access to a CUDA environment that can support `flash-linear-attention` and `FlashFFTConv`

At that point, the cleanest change is to preserve the existing public interfaces and replace only the sequence-mixing internals.

## Still Missing Vs. The Paper

Even after the current stair-only and augmentation updates, several parts of the paper recipe are still not fully matched:

- exact `FlashFFTConv` long-convolution backend
- exact `fla.layers.DeltaNet` backend in a validated CUDA environment
- per-sequence min-max normalization used by the paper, instead of our dataset-level `[-1, 1] -> [0, 1]` mapping
- the paper’s long-sequence setup with `L=2048`, `p=48`, and autoregressive chunk rollout
- inference-time flip equivariance
- FFT-based inference-time downsampling for long seasonal signals
- synthetic data generation (`KernelSynth`, spike processes, TSI)
- the paper’s training schedule details such as WSD

So the current code is best understood as:

- a UCI HAR-specific Reverso-style baseline
- narrowed to configurable multivariate IMU forecasting
- with paper-inspired augmentation and optional future hooks for the CUDA backends
