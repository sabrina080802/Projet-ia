"""Stage 5 - Evaluation.

Evaluate trained weights on the held-out test split with ``model.val()`` and
return the headline detection metrics.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from config import settings
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate(weights: str | Path, data_yaml: str | Path) -> dict[str, float]:
    """Run validation on the test split and return key metrics.

    Args:
        weights: Path to the trained ``best.pt``.
        data_yaml: Path to the dataset ``data.yaml`` (must define a ``test`` split).

    Returns:
        Mapping with ``map50``, ``map50_95``, ``precision`` and ``recall``.
    """
    logger.info("Evaluating %s on the test split", weights)
    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split="test", imgsz=settings.train.imgsz)

    summary = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    logger.info("Test metrics: %s", summary)
    return summary
