"""Stage 1 - Extraction.

Resolve the source dataset from the Ultralytics Platform and materialise it in
the raw data directory so the downstream stages (validation, preparation) can
work on local files. The dataset is referenced by its ``ul://`` URI; Ultralytics
downloads and caches it on first use, and we copy the resolved YOLO tree
(``images/`` + ``labels/``) into ``raw_dir`` so the run stays self-contained.

A local directory path is also accepted and used as-is, which is handy when the
dataset has already been pulled or is being iterated on offline.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from config import settings
from sharp import platform
from sharp.logging_utils import get_logger

logger = get_logger(__name__)


def _resolve_platform_dataset(uri: str) -> Path:
    """Download/cache a ``ul://`` dataset via ultralytics and return its root."""
    from ultralytics.data.utils import check_det_dataset

    platform.configure_ultralytics_auth()
    logger.info("Resolving Platform dataset %s", uri)
    info = check_det_dataset(uri)
    root = info.get("path")
    if not root:
        raise RuntimeError(f"Ultralytics returned no local path for dataset {uri}: {info}")
    return Path(root)


def extract(dataset: str | None = None, raw_dir: str | Path | None = None) -> Path:
    """Materialise the source dataset into ``raw_dir``.

    Args:
        dataset: A dataset slug, a full ``ul://`` URI, or a local dataset
            directory. Defaults to ``settings.ultralytics.dataset`` (built into
            ``ul://<username>/datasets/<dataset>``).
        raw_dir: Destination directory; defaults to ``settings.data.raw_dir``.

    Returns:
        Path to the directory holding the extracted images and labels.

    Raises:
        RuntimeError: If the dataset cannot be resolved.
    """
    raw_path = Path(raw_dir or settings.data.raw_dir)

    # A local directory is used in place; otherwise resolve the ul:// dataset.
    if dataset and Path(dataset).is_dir():
        source = Path(dataset)
        logger.info("Using local dataset directory %s", source)
    else:
        source = _resolve_platform_dataset(platform.dataset_uri(dataset))

    if source.resolve() == raw_path.resolve():
        logger.info("Dataset already at %s; nothing to copy.", raw_path)
        return raw_path

    if raw_path.exists():
        shutil.rmtree(raw_path)
    logger.info("Copying dataset %s -> %s", source, raw_path)
    shutil.copytree(source, raw_path)
    logger.info("Extraction complete: %s", raw_path)
    return raw_path


def clean(raw_dir: str | Path | None = None) -> None:
    """Remove a previously extracted raw dataset directory."""
    raw_path = Path(raw_dir or settings.data.raw_dir)
    if raw_path.exists():
        shutil.rmtree(raw_path)
        logger.info("Removed %s", raw_path)
