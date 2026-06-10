"""Cloud stage - dataset upload.

Push a local YOLO dataset archive to Ultralytics HUB so it can be used as the
source for cloud training. The archive must follow the YOLO structure with a
``data.yaml`` at its root (the local preparation stage produces exactly that).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from config import settings
from sharp.cloud.client import hub_client
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


def zip_dataset(yolo_dir: str | Path | None = None) -> Path:
    """Zip a prepared YOLO dataset directory for upload.

    Args:
        yolo_dir: Prepared dataset root; defaults to ``settings.data.yolo_dir``.

    Returns:
        Path to the created ``.zip`` archive.
    """
    yolo_path = Path(yolo_dir or settings.data.yolo_dir)
    if not (yolo_path / "data.yaml").exists():
        raise RuntimeError(f"{yolo_path} has no data.yaml; run the preparation stage first.")
    archive = shutil.make_archive(str(yolo_path), "zip", root_dir=yolo_path)
    logger.info("Created dataset archive %s", archive)
    return Path(archive)


def upload(archive: str | Path, name: str = "sharp-hands") -> str:
    """Create a HUB dataset entry and upload the archive to it.

    Args:
        archive: Path to the zipped YOLO dataset.
        name: Display name for the dataset on HUB.

    Returns:
        The new HUB dataset id.
    """
    client = hub_client()
    dataset = client.dataset()
    dataset.create_dataset({"meta": {"name": name}})
    dataset_id = dataset.id
    logger.info("Created HUB dataset %s (%s); uploading archive", name, dataset_id)
    dataset.upload_dataset(file=str(archive))
    logger.info("Upload complete for dataset %s", dataset_id)
    return dataset_id
