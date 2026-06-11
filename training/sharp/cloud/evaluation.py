"""Cloud stage - training status and weight download from the Platform.

After a run, pull the metrics the Platform recorded and download the best
weights so they can be served locally (the serving backend's ``best.pt``) or
evaluated offline.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Any

from config import settings
from sharp import platform
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


def fetch_metrics(model_id: str | None = None) -> dict[str, Any]:
    """Return the Platform training status and metrics for a model.

    Args:
        model_id: Model id; defaults to ``settings.ultralytics.model_id``.

    Returns:
        The raw ``GET /api/models/{id}/training`` payload (status + metrics).

    Raises:
        RuntimeError: If no model id is configured.

    Note:
        This route only accepts API-key auth for **public** projects. For a
        private project it requires a browser session, so polling here will fail
        — watch the run on the Platform dashboard instead.
    """
    model_id = model_id or settings.ultralytics.model_id
    if not model_id:
        raise RuntimeError("No model id configured (set ultralytics.model_id or pass --model-id).")
    response = platform.platform_client().get(f"models/{model_id}/training")
    logger.info("Fetched training status for model %s", model_id)
    return response


def _pick_weight_url(files: Any, kind: str) -> str | None:
    """Best-effort extraction of a weights URL from a ``/files`` response.

    The response schema is not documented, so handle the plausible shapes: a list
    of file objects, a ``{"files"|"data"|"results": [...]}`` envelope, or a flat
    ``{filename: url}`` mapping. Prefer a file whose name contains ``kind``
    (``best``/``last``), else the first ``.pt``.
    """

    def url_of(item: Any) -> str | None:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("url", "downloadUrl", "signedUrl", "href", "link"):
                if isinstance(item.get(key), str):
                    return item[key]
        return None

    def name_of(item: Any) -> str:
        if isinstance(item, dict):
            for key in ("name", "filename", "path", "key"):
                if isinstance(item.get(key), str):
                    return item[key]
        return ""

    items: list[Any] = []
    if isinstance(files, list):
        items = files
    elif isinstance(files, dict):
        for key in ("files", "data", "results"):
            if isinstance(files.get(key), list):
                items = files[key]
                break
        else:
            items = [{"name": k, "url": v} for k, v in files.items() if isinstance(v, str)]

    for item in items:
        if kind in name_of(item).lower():
            return url_of(item)
    for item in items:
        if name_of(item).lower().endswith(".pt"):
            return url_of(item)
    return None


def _download_via_uri(dest: Path) -> Path:
    """Fallback: resolve the model's ``ul://`` URI and copy its checkpoint."""
    from ultralytics import YOLO

    platform.configure_ultralytics_auth()
    uri = platform.model_uri()
    logger.info("Resolving %s through ultralytics to obtain the weights", uri)
    model = YOLO(uri)
    ckpt = getattr(model, "ckpt_path", None)
    if ckpt and Path(ckpt).exists():
        shutil.copy2(ckpt, dest)
        logger.info("Copied %s -> %s", ckpt, dest)
        return dest
    raise RuntimeError(
        "Downloaded the model but could not locate its .pt on disk. Download it "
        "from the Platform UI (Model -> Overview -> Download) instead."
    )


def download_weights(
    model_id: str | None = None,
    dest: str | Path = "runs/cloud/best.pt",
    kind: str = "best",
) -> Path:
    """Download trained weights from the Platform.

    Tries the REST files endpoint first (signed URLs); if it is unavailable or
    its shape is unexpected, falls back to resolving the model's ``ul://`` URI
    through the ``ultralytics`` package.

    Args:
        model_id: Model id; defaults to ``settings.ultralytics.model_id``.
        dest: Local path to write the weights to.
        kind: Which checkpoint to prefer (``best`` or ``last``).

    Returns:
        Path to the downloaded weights file.
    """
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    model_id = model_id or settings.ultralytics.model_id

    if model_id:
        try:
            files = platform.platform_client().get(f"models/{model_id}/files")
            url = _pick_weight_url(files, kind)
        except Exception as exc:  # noqa: BLE001 - any REST failure -> ul:// fallback
            logger.warning("REST files lookup failed (%s); falling back to ul:// resolution.", exc)
            url = None
        if url:
            logger.info("Downloading %s weights to %s", kind, dest_path)
            urllib.request.urlretrieve(url, dest_path)  # noqa: S310 (signed Platform URL)
            return dest_path

    return _download_via_uri(dest_path)
