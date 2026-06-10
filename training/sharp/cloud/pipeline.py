"""Orchestrator for the cloud pipeline.

Prepares a local YOLO dataset, uploads it to Ultralytics HUB, and registers a
HUB model with the training config. Cloud GPU training itself is launched from
the HUB UI (or via ``train.train_programmatic``); afterwards
``cloud.evaluation`` retrieves metrics and downloads the best weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from sharp.cloud import dataset as cloud_dataset
from sharp.cloud import train as cloud_train
from sharp.local import extraction, preparation, validation
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CloudSetupResult:
    """Identifiers produced while setting up cloud training."""

    dataset_id: str
    model_id: str
    model_url: str


def setup(dataset_id: str | None = None, reuse_remote: bool = False) -> CloudSetupResult:
    """Prepare and register everything needed for HUB cloud training.

    Args:
        dataset_id: Existing HUB dataset id to reuse as the source. When None and
            ``reuse_remote`` is False, the dataset is extracted, prepared locally
            and re-uploaded as a clean YOLO archive.
        reuse_remote: When True, skip local prep/upload and train directly off the
            ``dataset_id`` already on HUB.

    Returns:
        A :class:`CloudSetupResult` with the dataset and model identifiers.
    """
    if reuse_remote:
        if not dataset_id:
            raise RuntimeError("reuse_remote=True requires a dataset_id.")
        hub_dataset_id = dataset_id
    else:
        logger.info("Preparing local YOLO dataset for upload")
        raw_dir = extraction.extract(dataset_id=dataset_id)
        report = validation.validate(raw_dir)
        if not report.ok:
            logger.warning("Validation issues found; uploading anyway.")
        preparation.prepare(raw_dir=raw_dir)
        archive = cloud_dataset.zip_dataset()
        hub_dataset_id = cloud_dataset.upload(archive)

    model_id = cloud_train.create_model(dataset_id=hub_dataset_id)
    model_url = f"https://hub.ultralytics.com/models/{model_id}"
    logger.info("Cloud setup complete. Launch training at %s", model_url)
    return CloudSetupResult(dataset_id=hub_dataset_id, model_id=model_id, model_url=model_url)
