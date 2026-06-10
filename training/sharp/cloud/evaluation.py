"""Cloud stage - metrics retrieval and weight download.

Pull the training metrics HUB recorded for a model and download its best
weights so they can be served locally or evaluated offline.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

from config import settings
from sharp.cloud.client import hub_client
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


def fetch_metrics(model_id: str | None = None) -> dict[str, Any]:
    """Return the metrics HUB stored for a model.

    Args:
        model_id: HUB model id; defaults to ``settings.ultralytics.model_id``.

    Returns:
        The metrics payload as returned by the HUB SDK.
    """
    model_id = model_id or settings.ultralytics.model_id
    if not model_id:
        raise RuntimeError("No model id configured.")
    model = hub_client().model(model_id)
    metrics = model.get_metrics()
    logger.info("Fetched metrics for model %s", model_id)
    return metrics


def download_weights(
    model_id: str | None = None,
    kind: str = "best",
    dest: str | Path = "runs/cloud/best.pt",
) -> Path:
    """Download trained weights from HUB.

    Args:
        model_id: HUB model id; defaults to ``settings.ultralytics.model_id``.
        kind: Which checkpoint to fetch (``best`` or ``last``).
        dest: Local path to write the weights to.

    Returns:
        Path to the downloaded weights file.
    """
    model_id = model_id or settings.ultralytics.model_id
    if not model_id:
        raise RuntimeError("No model id configured.")

    model = hub_client().model(model_id)
    url = model.get_weights_url(kind)
    if not url:
        raise RuntimeError(f"HUB returned no {kind} weights URL for model {model_id}.")

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s weights to %s", kind, dest_path)
    urllib.request.urlretrieve(url, dest_path)  # noqa: S310 (trusted HUB URL)
    return dest_path
