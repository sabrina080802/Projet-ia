"""SHARP serving backend.

FastAPI service that loads the trained YOLO finger-detection model **once at
startup** and exposes a ``/predict`` endpoint. The JSON response intentionally
mirrors the Ultralytics HUB hosted-inference shape
(``{"images": [{"results": [...]}]}``) so the existing web dashboard consumes it
without any change.

Configuration is read from environment variables (see :class:`Settings`):

* ``SHARP_MODEL_PATH`` - local path to the weights (default ``/models/best.pt``).
* ``SHARP_MODEL_URL``  - Ultralytics HUB model URL to download if the local file
  is absent (requires ``SHARP_API_KEY``).
* ``SHARP_API_KEY``    - optional. When set, it is required as the ``x-api-key``
  header on ``/predict`` and is used to authenticate the HUB download.
* ``SHARP_CONF`` / ``SHARP_IOU`` / ``SHARP_IMGSZ`` - inference hyper-parameters.
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


class Settings:
    """Runtime configuration sourced from environment variables."""

    def __init__(self) -> None:
        """Read the serving configuration from the environment."""
        self.model_path: str = os.getenv("SHARP_MODEL_PATH", "/models/best.pt")
        self.model_url: str = os.getenv("SHARP_MODEL_URL", "")
        self.api_key: str = os.getenv("SHARP_API_KEY", "")
        self.conf: float = float(os.getenv("SHARP_CONF", "0.25"))
        self.iou: float = float(os.getenv("SHARP_IOU", "0.45"))
        self.imgsz: int = int(os.getenv("SHARP_IMGSZ", "640"))


settings = Settings()


def load_model() -> YOLO:
    """Load the YOLO model from the local path, falling back to an HUB download.

    Returns:
        A ready-to-use :class:`~ultralytics.YOLO` model.

    Raises:
        RuntimeError: If no local weights exist and no HUB URL is configured.
    """
    if Path(settings.model_path).exists():
        return YOLO(settings.model_path)

    if settings.model_url:
        if settings.api_key:
            from ultralytics import hub

            hub.login(settings.api_key)
        return YOLO(settings.model_url)

    raise RuntimeError(
        f"No model found at {settings.model_path!r} and SHARP_MODEL_URL is unset. "
        "Place best.pt in the mounted models directory or set SHARP_MODEL_URL."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201 - FastAPI lifespan signature
    """Load the model before the API starts serving requests."""
    app.state.model = load_model()
    yield


app = FastAPI(title="SHARP serving API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_auth(provided_key: str | None) -> None:
    """Reject the request when an API key is configured but not supplied.

    Args:
        provided_key: Value of the incoming ``x-api-key`` header, if any.

    Raises:
        HTTPException: ``401`` when the configured key does not match.
    """
    if settings.api_key and provided_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing x-api-key.")


def _to_hub_payload(results: Any) -> dict[str, Any]:
    """Convert Ultralytics results into the HUB hosted-inference JSON shape.

    Args:
        results: The list returned by ``model.predict``.

    Returns:
        A ``{"images": [{"results": [...]}]}`` dict where each result carries the
        class ``name``, numeric ``class`` id, ``confidence`` and an ``x1/y1/x2/y2``
        ``box`` expressed in pixels of the submitted frame.
    """
    result = results[0]
    names: dict[int, str] = result.names
    detections: list[dict[str, Any]] = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        detections.append(
            {
                "name": names.get(class_id, str(class_id)),
                "class": class_id,
                "confidence": float(box.conf[0]),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )
    return {"images": [{"results": detections}]}


@app.get("/health")
def health() -> dict[str, Any]:
    """Report liveness and whether the model is loaded."""
    return {"status": "ok", "model_loaded": getattr(app.state, "model", None) is not None}


@app.post("/predict")
async def predict(
    file: Annotated[UploadFile, File(...)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Run detection on a single uploaded frame.

    Args:
        file: Multipart image field (JPEG/PNG) sent by the dashboard.
        x_api_key: Optional API key header, validated when one is configured.

    Returns:
        Detections in the HUB hosted-inference JSON shape.

    Raises:
        HTTPException: ``400`` for an unreadable image, ``401`` for a bad key.
    """
    _check_auth(x_api_key)
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Unreadable image.") from exc

    results = app.state.model.predict(
        image,
        conf=settings.conf,
        iou=settings.iou,
        imgsz=settings.imgsz,
        verbose=False,
    )
    return _to_hub_payload(results)
