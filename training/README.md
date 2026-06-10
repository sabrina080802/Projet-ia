# SHARP — Training pipelines

Two interchangeable ways to train the finger-detection model, sharing the same
config, classes and dataset:

| Pipeline | Where it runs | Use it when |
|----------|---------------|-------------|
| **`cloud`** | Ultralytics Platform | Local training is too slow — train on the Platform (cloud GPU or streamed local) and pull the weights. |
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
│   └── cloud/   train (cloud-GPU + streamed local) → evaluation → inference
│   platform.py  shared Platform auth, REST client and ul:// URI builders
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
`ultralytics.username` / `dataset` / `project` / `model` (the slugs that build
your `ul://` references — copy them from the dataset/model pages on the
Platform), `train.model`, `train.epochs`, the 60/20/20 split and seed `42`. The
Platform API key (`ul_…`) goes in `.secrets.toml` (gitignored) or
`SHARP_ULTRALYTICS__API_KEY`.

## Local pipeline

Full run (extract → validate → prepare → train → evaluate):

```bash
python run.py local --dataset <DATASET_SLUG_OR_ul://_URI>
```

Individual stages:

```bash
python run.py local --stage extract  --dataset <DATASET_SLUG_OR_ul://_URI>
python run.py local --stage validate --raw-dir data/raw
python run.py local --stage prepare                       # 60/20/20, seed 42, writes data.yaml
python run.py local --stage train    --data-yaml data/yolo/data.yaml
python run.py local --stage evaluate --weights runs/sharp/weights/best.pt --data-yaml data/yolo/data.yaml
```

Metrics and loss curves are written by Ultralytics under `runs/` for the oral.

## Cloud pipeline (Ultralytics Platform)

The dataset already lives on the Platform and is referenced by its `ul://` URI
(`ul://<username>/datasets/<dataset>`), so there is nothing to upload — set the
slugs in `settings.toml`. Two ways to train:

```bash
# Train on THIS machine, streaming metrics + weights to the Platform (free).
# Creates the project/model on the Platform automatically.
python run.py cloud train

# Or dispatch a real cloud-GPU run via the REST API (needs a paid GPU plan).
# `setup` first provisions the project + model the REST job needs:
python run.py cloud setup                       # -> prints project_id / model_id
python run.py cloud train --cloud-gpu --gpu-type rtx-pro-6000

# After training: pull status/metrics / download best weights
python run.py cloud metrics  --model-id <MODEL_ID>
python run.py cloud download --model-id <MODEL_ID> --dest runs/cloud/best.pt

# Hosted inference on a single image (matches the web dashboard)
python run.py cloud predict path/to/image.jpg
```

> **Note on the two modes:** `cloud train` runs `model.train(project=<user>/<proj>)`
> via the `ultralytics` package — computation happens wherever you run it, metrics
> stream to the Platform dashboard, no subscription needed. `cloud train --cloud-gpu`
> calls `POST /api/training/start` to run on Ultralytics' GPUs (paid). `metrics`
> polls `GET /api/models/{id}/training`, which only accepts API-key auth for
> **public** projects — for a private project, watch the run on the dashboard.

## Quality & tests

```bash
ruff check . && black --check . && pytest
```

`ruff`, `black` and `pytest` also run via pre-commit and in CI on every PR.
