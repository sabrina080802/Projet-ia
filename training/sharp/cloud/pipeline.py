"""Orchestrator for the cloud pipeline.

The dataset already lives on the Ultralytics Platform and is referenced by its
``ul://`` URI, so there is nothing to upload. ``setup`` only provisions the
Platform project + model that a cloud-GPU run needs; afterwards launch training
with ``cloud train`` (local + streaming) or ``cloud train --cloud-gpu`` (REST),
then ``cloud.evaluation`` pulls metrics and downloads the best weights.

Plain ``cloud train`` (local streaming) creates the project/model on the fly, so
it does not require ``setup`` first — ``setup`` is for the REST cloud-GPU path.
"""

from __future__ import annotations

from dataclasses import dataclass

from sharp import platform
from sharp.cloud import train as cloud_train
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CloudSetupResult:
    """Identifiers/URIs produced while provisioning cloud training."""

    dataset_uri: str
    project_id: str
    model_id: str
    model_uri: str


def setup() -> CloudSetupResult:
    """Provision the Platform project + model for the configured ``ul://`` dataset.

    Returns:
        A :class:`CloudSetupResult` with the dataset URI, the REST project/model
        ids (needed by ``train_cloud``) and the model ``ul://`` URI.
    """
    dataset_uri = platform.dataset_uri()
    logger.info("Cloud training will read dataset %s", dataset_uri)

    project_id = cloud_train.ensure_project()
    model_id = cloud_train.create_model(project_id=project_id)
    model_uri = platform.model_uri()

    logger.info(
        "Cloud setup complete. project_id=%s model_id=%s model=%s", project_id, model_id, model_uri
    )
    logger.info(
        "Next: `cloud train` (local + stream) or `cloud train --cloud-gpu` (REST GPU). "
        "Save model_id=%s in settings.toml to reuse it.",
        model_id,
    )
    return CloudSetupResult(
        dataset_uri=dataset_uri,
        project_id=project_id,
        model_id=model_id,
        model_uri=model_uri,
    )
