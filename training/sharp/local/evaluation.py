"""Stage 5 - Evaluation.

Evaluate trained weights on the held-out test split with ``model.val()`` and
return the headline detection metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

from config import settings
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate(weights: str | Path, data_yaml: str | Path) -> dict[str, float]:
    """Run validation on the test split, persist and return key metrics.

    The headline metrics are written to ``test_metrics.json`` inside the run
    directory so the results survive (and can be shown at the oral) instead of
    only being printed.

    Args:
        weights: Path to the trained ``best.pt``.
        data_yaml: Path to the dataset ``data.yaml`` (must define a ``test`` split).

    Returns:
        Mapping with ``map50``, ``map50_95``, ``precision``, ``recall``, the
        per-image inference time and the derived FPS.
    """
    logger.info("Evaluating %s on the test split", weights)
    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split="test", imgsz=settings.train.imgsz)

    inference_ms = float(metrics.speed.get("inference", 0.0))
    summary = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "inference_ms": inference_ms,
        "fps": round(1000.0 / inference_ms, 1) if inference_ms else 0.0,
    }
    logger.info("Test metrics: %s", summary)

    out_path = Path(metrics.save_dir) / "test_metrics.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return summary
