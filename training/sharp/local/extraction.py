"""Stage 1 - Extraction.

Download the annotated dataset from Ultralytics HUB using the official
``hub_sdk`` and unpack it into the raw data directory. The dataset id is
configurable via ``settings.toml`` or the ``--dataset-id`` CLI argument.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

from hub_sdk import HUBClient

from config import settings
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


def _client() -> HUBClient:
    """Build an authenticated HUB client from the configured API key."""
    api_key = settings.ultralytics.api_key
    if not api_key or api_key == "YOUR_ULTRALYTICS_API_KEY":
        raise RuntimeError(
            "Missing Ultralytics API key. Copy .secrets.toml.example to .secrets.toml "
            "and set ultralytics.api_key (or export SHARP_ULTRALYTICS__API_KEY)."
        )
    return HUBClient({"api_key": api_key})


def extract(dataset_id: str | None = None, raw_dir: str | Path | None = None) -> Path:
    """Download and unzip the HUB dataset into ``raw_dir``.

    Args:
        dataset_id: HUB dataset id; defaults to ``settings.ultralytics.dataset_id``.
        raw_dir: Destination directory; defaults to ``settings.data.raw_dir``.

    Returns:
        Path to the directory holding the extracted images and annotations.

    Raises:
        RuntimeError: If the API key or dataset id is missing, or the download fails.
    """
    dataset_id = dataset_id or settings.ultralytics.dataset_id
    if not dataset_id or dataset_id == "YOUR_DATASET_ID":
        raise RuntimeError(
            "No dataset id configured. Set ultralytics.dataset_id in settings.toml "
            "or pass --dataset-id."
        )

    raw_path = Path(raw_dir or settings.data.raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    logger.info("Requesting download link for HUB dataset %s", dataset_id)
    dataset = _client().dataset(dataset_id)
    download_url = dataset.get_download_link()
    if not download_url:
        raise RuntimeError(f"HUB returned no download link for dataset {dataset_id}.")

    archive_path = raw_path / "dataset.zip"
    logger.info("Downloading dataset archive to %s", archive_path)
    urllib.request.urlretrieve(download_url, archive_path)  # noqa: S310 (trusted HUB URL)

    logger.info("Unzipping archive into %s", raw_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(raw_path)
    archive_path.unlink(missing_ok=True)

    logger.info("Extraction complete: %s", raw_path)
    return raw_path


def clean(raw_dir: str | Path | None = None) -> None:
    """Remove a previously extracted raw dataset directory."""
    raw_path = Path(raw_dir or settings.data.raw_dir)
    if raw_path.exists():
        shutil.rmtree(raw_path)
        logger.info("Removed %s", raw_path)
