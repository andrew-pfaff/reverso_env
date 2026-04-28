# Reverso

Machine learning project scaffold using a `src`-layout Python package and a local virtual environment.

## Project Layout

```text
.
├── README.md
├── pyproject.toml
├── src/
│   └── reverso/
│       ├── data/
│       ├── models/
│       ├── training/
│       └── utils/
└── data/
```

## Environment Setup

Create the virtual environment in the project root:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade `pip` if needed:

```bash
python -m pip install --upgrade pip
```

Install the project and dependencies in editable mode:

```bash
python -m pip install -e .
```

This installs:

- `reverso` as an editable package from `src/`
- `numpy`
- `torch`
- `matplotlib`
- `ipykernel`

## Jupyter Notebook Kernel

Register the virtual environment as a notebook kernel:

```bash
python -m ipykernel install --user --name reverso --display-name Reverso
```

After that, select `Reverso` inside Jupyter Notebook or JupyterLab.

To confirm the kernel is installed:

```bash
jupyter kernelspec list
```

## Verification

Verify the package and core dependencies import correctly:

```bash
python -c "import reverso, numpy, torch, matplotlib; print('ok')"
```

## Daily Use

Each new shell session:

```bash
source .venv/bin/activate
```

If you are working in notebooks, make sure the selected kernel is `Reverso`.

## Notes

- The package name is `reverso`.
- Source code lives under `src/reverso`.
- Raw or generated data can go under `data/`.
