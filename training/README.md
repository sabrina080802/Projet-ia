# SHARP — Training pipelines

Two interchangeable ways to train the finger-detection model, sharing the same
config, classes and dataset:

| Pipeline | Where it runs | Use it when |
|----------|---------------|-------------|
| **`cloud`** | Ultralytics HUB | Local training is too slow — upload the dataset, train on HUB, pull the weights. |
| **`local`** | Your machine (CUDA / Apple MPS / CPU) | You want the full, reproducible ML pipeline required by the brief. |

Both produce a YOLO11 model and use the six fixed classes
`0_doigt … 5_doigts`.

## Layout

```
training/
├── config.py / settings.toml / .secrets.toml   # dynaconf configuration
├── run.py                                       # CLI (local / cloud)
├── sharp/
│   ├── classes.py, logging_utils.py             # shared
│   ├── local/   extraction → validation → preparation → training → evaluation
│   └── cloud/   dataset → train → evaluation → inference
└── tests/                                        # unit tests (run in CI)
```

## Setup

```bash
cd training
python -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -e ".[dev]"
pre-commit install

cp .secrets.toml.example .secrets.toml              # then add your API key
```

Configure `settings.toml` (or `SHARP_*` env vars). Key values:
`ultralytics.dataset_id`, `train.model`, `train.epochs`, the 60/20/20 split and
seed `42`. The Ultralytics API key goes in `.secrets.toml` (gitignored) or
`SHARP_ULTRALYTICS__API_KEY`.

## Local pipeline

Full run (extract → validate → prepare → train → evaluate):

```bash
python run.py local --dataset-id <HUB_DATASET_ID>
```

Individual stages:

```bash
python run.py local --stage extract  --dataset-id <HUB_DATASET_ID>
python run.py local --stage validate --raw-dir data/raw
python run.py local --stage prepare                       # 60/20/20, seed 42, writes data.yaml
python run.py local --stage train    --data-yaml data/yolo/data.yaml
python run.py local --stage evaluate --weights runs/sharp/weights/best.pt --data-yaml data/yolo/data.yaml
```

Metrics and loss curves are written by Ultralytics under `runs/` for the oral.

## Cloud pipeline (Ultralytics HUB)

```bash
# Prepare + upload the dataset and register a HUB model with the training config
python run.py cloud setup --dataset-id <HUB_DATASET_ID>

# Launch Cloud GPU training from the printed HUB model URL (Pro feature),
# or run it programmatically (streams metrics/weights to HUB):
python run.py cloud train --model-id <HUB_MODEL_ID>

# After training: pull metrics / download best weights
python run.py cloud metrics  --model-id <HUB_MODEL_ID>
python run.py cloud download --model-id <HUB_MODEL_ID> --dest runs/cloud/best.pt

# Hosted inference on a single image (matches the web dashboard)
python run.py cloud predict path/to/image.jpg
```

> **Note on "cloud GPU":** Ultralytics Cloud Training runs on Ultralytics' GPUs
> and is a Pro feature launched from the HUB UI. `cloud setup` does everything
> programmatically up to that click; `cloud train` is the no-subscription
> fallback that trains via the `ultralytics` package while syncing to HUB.

## Quality & tests

```bash
ruff check . && black --check . && pytest
```

`ruff`, `black` and `pytest` also run via pre-commit and in CI on every PR.
