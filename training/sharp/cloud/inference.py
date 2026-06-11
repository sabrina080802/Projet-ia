"""Cloud stage - hosted inference on the Ultralytics Platform.

Thin client over the Platform's hosted-prediction endpoint
(``POST /api/models/{modelId}/predict``). Sends an image to the trained model and
normalizes the response into detections plus the summed finger count, mirroring
what the web dashboard displays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import settings
from sharp import platform
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
        RuntimeError: If the API key or model id is missing, or the API errors.
    """
    model_id = settings.ultralytics.model_id
    if not model_id:
        raise RuntimeError(
            "Hosted inference needs ultralytics.model_id. Run `cloud setup`, train "
            "the model, then set ultralytics.model_id (or pass --model-id)."
        )

    client = platform.platform_client()
    data = {
        "imgsz": settings.infer.imgsz,
        "conf": settings.infer.conf,
        "iou": settings.infer.iou,
    }
    logger.info("POST models/%s/predict", model_id)
    with open(image_path, "rb") as handle:
        response = client.session.post(
            client.url(f"models/{model_id}/predict"),
            data=data,
            files={"file": handle},
            timeout=30,
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Platform predict failed ({response.status_code}): {response.text[:500]}"
        )
    return _normalize(response.json())


def _iter_raw_detections(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull the per-detection dicts out of a hosted-inference payload.

    Tolerates the documented HUB-style ``{"images": [{"results": [...]}]}`` shape
    as well as a flat ``{"predictions"|"results"|"boxes": [...]}`` envelope.
    """
    if isinstance(payload.get("images"), list):
        detections: list[dict[str, Any]] = []
        for image in payload["images"]:
            detections.extend(image.get("results", []) or [])
        return detections
    for key in ("predictions", "results", "detections", "boxes"):
        if isinstance(payload.get(key), list):
            return payload[key]
    return []


def _normalize(payload: dict[str, Any]) -> InferenceResult:
    """Convert a raw hosted-inference payload into an :class:`InferenceResult`."""
    detections: list[Detection] = []
    for result in _iter_raw_detections(payload):
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
