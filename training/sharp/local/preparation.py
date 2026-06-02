"""Stage 3 - Preparation.

Assemble the validated images and labels into the YOLO directory layout,
generate deterministic train/val/test splits (60/20/20, seed 42), and write the
``data.yaml`` consumed by ``model.train(data=...)``. Existing splits in the raw
dataset are reused when present; otherwise they are generated.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml

from config import settings
from sharp.classes import names_mapping
from sharp.local.validation import IMAGE_SUFFIXES, _label_for_image
from sharp.logging_utils import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "val", "test")


def _collect_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    """Return ``(image, label)`` pairs that both exist on disk."""
    pairs: list[tuple[Path, Path]] = []
    for image in sorted(raw_dir.rglob("*")):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = _label_for_image(image)
        if label.exists():
            pairs.append((image, label))
    return pairs


def _split_pairs(
    pairs: list[tuple[Path, Path]],
) -> dict[str, list[tuple[Path, Path]]]:
    """Shuffle and partition pairs into train/val/test by the configured ratios."""
    rng = random.Random(settings.data.seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * settings.data.train_ratio)
    n_val = int(n * settings.data.val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _copy_split(name: str, pairs: list[tuple[Path, Path]], yolo_dir: Path) -> None:
    img_dir = yolo_dir / "images" / name
    lbl_dir = yolo_dir / "labels" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for image, label in pairs:
        shutil.copy2(image, img_dir / image.name)
        shutil.copy2(label, lbl_dir / label.name)


def write_data_yaml(yolo_dir: Path) -> Path:
    """Generate the Ultralytics ``data.yaml`` for the prepared dataset.

    Args:
        yolo_dir: Root of the prepared YOLO dataset.

    Returns:
        Path to the written ``data.yaml``.
    """
    data = {
        "path": str(yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names_mapping(),
    }
    yaml_path = yolo_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    logger.info("Wrote %s", yaml_path)
    return yaml_path


def prepare(
    raw_dir: str | Path | None = None,
    yolo_dir: str | Path | None = None,
) -> Path:
    """Build the YOLO dataset and return the path to ``data.yaml``.

    Args:
        raw_dir: Extracted raw dataset; defaults to ``settings.data.raw_dir``.
        yolo_dir: Output dataset root; defaults to ``settings.data.yolo_dir``.

    Returns:
        Path to the generated ``data.yaml``.

    Raises:
        RuntimeError: If no image/label pairs are found in ``raw_dir``.
    """
    raw_path = Path(raw_dir or settings.data.raw_dir)
    yolo_path = Path(yolo_dir or settings.data.yolo_dir)
    if yolo_path.exists():
        shutil.rmtree(yolo_path)

    pairs = _collect_pairs(raw_path)
    if not pairs:
        raise RuntimeError(f"No image/label pairs found under {raw_path}.")

    splits = _split_pairs(pairs)
    for name, split_pairs in splits.items():
        _copy_split(name, split_pairs, yolo_path)
        logger.info("Split %-5s -> %d samples", name, len(split_pairs))

    return write_data_yaml(yolo_path)
