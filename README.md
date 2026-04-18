# FungiTastic Playground

This is a project for `Neural Network - Theory and Practice` course at the University of Wroclaw.

Dataset: https://bohemianvra.github.io/FungiTastic/

Goal: Fungi Segmentation

### Instalation

Install `uv` from https://docs.astral.sh/uv/.

```bash
uv sync
```

### Dataset download

```bash
uv run scripts/download.py --masks --images --size 300 --save_path ./data
```