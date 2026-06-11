"""SHARP serving backend.

FastAPI service that loads the trained YOLO finger-detection model **once at
startup** and exposes a ``/predict`` endpoint. The JSON response intentionally
mirrors the Ultralytics HUB hosted-inference shape
(``{"images": [{"results": [...]}]}``) so the existing web dashboard consumes it
without any change.

Configuration is read from environment variables (see :class:`Settings`):

* ``SHARP_MODEL_PATH`` - local path to the weights (default ``/models/best.pt``).
* ``SHARP_MODEL_URL``  - Ultralytics Platform model URI to download if the local
  file is absent (requires ``SHARP_API_KEY``). Use the ``ul://username/project/model``
  scheme; legacy ``https://hub.ultralytics.com/models/<id>`` URLs still work until
  HUB is wound down (end of July 2026).
* ``SHARP_API_KEY``    - optional. When set, it is required as the ``x-api-key``
  header on ``/predict`` and is used to authenticate the Platform download
  (a ``ul_``-prefixed Platform API key).
* ``SHARP_CONF`` / ``SHARP_IOU`` / ``SHARP_IMGSZ`` - inference hyper-parameters.
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import anyio
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


def _authenticate(api_key: str) -> None:
    """Register the Ultralytics Platform API key for ``ul://`` model downloads.

    Sets both the documented ``ULTRALYTICS_API_KEY`` environment variable and the
    persisted ``api_key`` setting (the programmatic equivalent of
    ``yolo settings api_key=...``), so authentication works headlessly in Docker
    regardless of which mechanism the installed ``ultralytics`` build reads.

    Args:
        api_key: A ``ul_``-prefixed Platform API key.
    """
    os.environ.setdefault("ULTRALYTICS_API_KEY", api_key)
    from ultralytics import settings as ul_settings

    ul_settings.update({"api_key": api_key})


def load_model() -> YOLO:
    """Load the YOLO model from the local path, falling back to a Platform download.

    Returns:
        A ready-to-use :class:`~ultralytics.YOLO` model.

    Raises:
        RuntimeError: If no local weights exist and no model URI is configured.
    """
    if Path(settings.model_path).exists():
        return YOLO(settings.model_path)

    if settings.model_url:
        if settings.api_key:
            _authenticate(settings.api_key)
        return YOLO(settings.model_url)

    raise RuntimeError(
        f"No model found at {settings.model_path!r} and SHARP_MODEL_URL is unset. "
        "Place best.pt in the mounted models directory or set SHARP_MODEL_URL "
        "to a ul://username/project/model URI."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201 - FastAPI lifespan signature
    """Load the model before the API starts serving requests."""
    model = load_model()
    app.state.model = model
    # Surface which device the model landed on so a missing GPU passthrough is
    # obvious in the logs (CPU inference is what caps throughput at a few FPS).
    device = getattr(model, "device", "unknown")
    print(f"[SHARP] Model loaded on device: {device}", flush=True)
    # Warm up so the first real request doesn't pay CUDA/graph init latency.
    warmup = Image.new("RGB", (settings.imgsz, settings.imgsz))
    model.predict(warmup, imgsz=settings.imgsz, verbose=False)
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

    # Run the blocking inference in a worker thread so the event loop stays free
    # to receive the next frame's upload while the GPU is busy on this one.
    results = await anyio.to_thread.run_sync(
        lambda: app.state.model.predict(
            image,
            conf=settings.conf,
            iou=settings.iou,
            imgsz=settings.imgsz,
            verbose=False,
        )
    )
    return _to_hub_payload(results)
