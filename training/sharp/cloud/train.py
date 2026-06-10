"""Cloud stage - model creation and training on Ultralytics HUB.

Creating a HUB model with a training config registers the job on the platform.
Ultralytics Cloud GPU training is a Pro feature normally launched from the HUB
web UI ("Train Model" -> "Ultralytics Cloud"); this module creates the model and
config programmatically, then exposes a fully-programmatic alternative that runs
the training via the ``ultralytics`` package while streaming metrics to HUB.
"""

from __future__ import annotations

from typing import Any

from config import settings
from sharp.cloud.client import hub_client
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


def create_model(
    dataset_id: str | None = None,
    project_id: str | None = None,
    name: str = "sharp-yolo11n",
    architecture: str = "yolo11n",
) -> str:
    """Create a HUB model wired to a dataset, with the configured hyper-parameters.

    Args:
        dataset_id: HUB dataset id; defaults to ``settings.ultralytics.dataset_id``.
        project_id: HUB project id; defaults to ``settings.ultralytics.project_id``.
        name: Display name for the model on HUB.
        architecture: Base YOLO architecture to train.

    Returns:
        The new HUB model id (also the handle used for cloud training in the UI).
    """
    dataset_id = dataset_id or settings.ultralytics.dataset_id
    project_id = project_id or settings.ultralytics.project_id or None

    config = {
        "batchSize": str(settings.train.batch),
        "cache": "ram",
        "device": "",
        "epochs": str(settings.train.epochs),
        "imageSize": str(settings.train.imgsz),
        "patience": str(settings.train.patience),
    }
    data: dict[str, Any] = {
        "meta": {"name": name},
        "datasetId": dataset_id,
        "config": config,
    }
    if project_id:
        data["projectId"] = project_id

    client = hub_client()
    model = client.model()
    model.create_model({**data, "architecture": architecture})
    logger.info("Created HUB model %s (id=%s)", name, model.id)
    logger.info(
        "Launch Cloud GPU training from https://hub.ultralytics.com/models/%s "
        "or call train_programmatic('%s').",
        model.id,
        model.id,
    )
    return model.id


def train_programmatic(model_id: str | None = None) -> str:
    """Run training for a HUB model via the ``ultralytics`` package.

    This streams metrics and weights back to HUB. Use it when you do not have a
    Cloud GPU subscription; computation then happens wherever this runs.

    Args:
        model_id: HUB model id; defaults to ``settings.ultralytics.model_id``.

    Returns:
        The HUB model URL the run is bound to.
    """
    from ultralytics import YOLO

    model_id = model_id or settings.ultralytics.model_id
    if not model_id:
        raise RuntimeError("No model id; create one with create_model() first.")

    model_url = f"https://hub.ultralytics.com/models/{model_id}"
    logger.info("Starting HUB-bound training for %s", model_url)
    YOLO(model_url).train()
    return model_url
