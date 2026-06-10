"""Cloud stage - hosted inference.

Thin client over the Ultralytics hosted inference API. Sends an image to the
deployed model and normalizes the response into detections plus the summed
finger count, mirroring what the web dashboard displays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from config import settings
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """A single predicted bounding box."""

    label: str
    confidence: float
    fingers: int
    box: dict[str, float]


@dataclass
class InferenceResult:
    """Normalized inference response."""

    detections: list[Detection]
    total_fingers: int
    raw: dict[str, Any]


def _fingers_from_label(label: str) -> int:
    """Parse the finger count from a class label such as ``3_doigts``."""
    head = label.split("_", 1)[0]
    return int(head) if head.isdigit() else 0


def predict(image_path: str | Path) -> InferenceResult:
    """Run hosted inference on a single image.

    Args:
        image_path: Path to the image to send.

    Returns:
        An :class:`InferenceResult` with detections and the total finger count.

    Raises:
        RuntimeError: If the API key or endpoint is missing.
        requests.HTTPError: If the API returns a non-2xx status.
    """
    api_key = settings.ultralytics.api_key
    if not api_key or api_key == "YOUR_ULTRALYTICS_API_KEY":
        raise RuntimeError("Missing Ultralytics API key for hosted inference.")

    endpoint = settings.ultralytics.inference_endpoint
    data = {
        "model": settings.ultralytics.inference_model_url,
        "imgsz": settings.infer.imgsz,
        "conf": settings.infer.conf,
        "iou": settings.infer.iou,
    }
    logger.info("POST %s (model=%s)", endpoint, data["model"])
    with open(image_path, "rb") as handle:
        response = requests.post(
            endpoint,
            headers={"x-api-key": api_key},
            data=data,
            files={"file": handle},
            timeout=30,
        )
    response.raise_for_status()
    payload = response.json()
    return _normalize(payload)


def _normalize(payload: dict[str, Any]) -> InferenceResult:
    """Convert a raw hosted-inference payload into an :class:`InferenceResult`."""
    detections: list[Detection] = []
    for image in payload.get("images", []):
        for result in image.get("results", []):
            label = str(result.get("name") or result.get("class") or result.get("label", ""))
            fingers = _fingers_from_label(label)
            detections.append(
                Detection(
                    label=label,
                    confidence=float(result.get("confidence", result.get("score", 0.0))),
                    fingers=fingers,
                    box=result.get("box", {}),
                )
            )
    total = sum(det.fingers for det in detections)
    logger.info("Detections: %d, total fingers: %d", len(detections), total)
    return InferenceResult(detections=detections, total_fingers=total, raw=payload)
