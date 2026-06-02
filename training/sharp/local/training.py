"""Stage 4 - Training.

Train a YOLO11 model locally with Ultralytics. Hyper-parameters and the model
size come from ``settings.toml`` and can be overridden per run. Ultralytics
writes loss/metric curves under ``runs/`` for the oral presentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from config import settings
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


def _resolve_device() -> str | None:
    """Return the configured device, or None to let Ultralytics auto-detect."""
    device = str(settings.train.device).strip()
    return device or None


def train(data_yaml: str | Path, **overrides: Any) -> Path:
    """Train a YOLO model and return the path to the best weights.

    Args:
        data_yaml: Path to the ``data.yaml`` produced by the preparation stage.
        **overrides: Optional Ultralytics ``train`` kwargs overriding settings
            (e.g. ``epochs=50``, ``model="yolo11s.pt"``).

    Returns:
        Path to ``best.pt`` for the trained run.
    """
    model_name = overrides.pop("model", settings.train.model)
    params: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": settings.train.epochs,
        "imgsz": settings.train.imgsz,
        "batch": settings.train.batch,
        "patience": settings.train.patience,
        "device": _resolve_device(),
        "project": settings.train.project,
        "name": settings.train.name,
        "seed": settings.data.seed,
    }
    params.update(overrides)

    logger.info("Training %s with %s", model_name, params)
    model = YOLO(model_name)
    results = model.train(**params)

    best = Path(results.save_dir) / "weights" / "best.pt"
    logger.info("Training finished. Best weights: %s", best)
    return best
