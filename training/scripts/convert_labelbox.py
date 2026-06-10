"""Convert a Labelbox NDJSON export into a YOLO dataset zip for Ultralytics HUB.

This is an optional, one-off data-preparation helper for the manual upload path
(Labelbox export -> YOLO layout -> ``dataset.zip`` dropped on Ultralytics HUB).
The local/cloud training pipelines download an already-YOLO dataset via the HUB
SDK and do not need this script.

The class names come from :func:`sharp.classes.names_mapping`, so the six fixed
classes (``0_doigt`` .. ``5_doigts``) stay the single source of truth instead of
being re-declared here.

Run it from the ``training/`` directory (with the package installed) so the
``sharp`` import resolves::

    python scripts/convert_labelbox.py --ndjson handsyolo26.ndjson
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import yaml

from sharp.classes import names_mapping
from sharp.logging_utils import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "val")


def _ensure_dirs(out_dir: Path) -> None:
    """Create the images/labels train/val directories under ``out_dir``."""
    for kind in ("images", "labels"):
        for split in SPLITS:
            (out_dir / kind / split).mkdir(parents=True, exist_ok=True)


def _write_dataset_yaml(out_dir: Path) -> None:
    """Write the Ultralytics ``dataset.yaml`` with the canonical class names."""
    data = {
        "path": "./",
        "train": "images/train",
        "val": "images/val",
        "names": names_mapping(),
    }
    with (out_dir / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    logger.info("Wrote %s", out_dir / "dataset.yaml")


def _process_image(record: dict, out_dir: Path) -> None:
    """Download one image and write its YOLO label file."""
    filename = record.get("file")
    if not filename:
        return
    split = record.get("split", "train")
    if split not in SPLITS:
        split = "train"

    image_path = out_dir / "images" / split / filename
    url = record.get("url")
    if url and not image_path.exists():
        logger.info("Downloading %s", filename)
        urllib.request.urlretrieve(url, image_path)

    label_path = out_dir / "labels" / split / (Path(filename).stem + ".txt")
    boxes = record.get("annotations", {}).get("boxes", [])
    lines = [f"{int(box[0])} {box[1]} {box[2]} {box[3]} {box[4]}" for box in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def convert(ndjson_path: Path, out_dir: Path) -> Path:
    """Convert an NDJSON export into a YOLO dataset and zip it.

    Args:
        ndjson_path: Path to the Labelbox/HUB NDJSON export.
        out_dir: Directory where the YOLO dataset is assembled.

    Returns:
        Path to the created ``dataset.zip``.
    """
    _ensure_dirs(out_dir)
    with ndjson_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "dataset":
                _write_dataset_yaml(out_dir)
            elif record.get("type") == "image":
                _process_image(record, out_dir)

    archive = shutil.make_archive(str(out_dir), "zip", str(out_dir))
    logger.info("Created %s - drop it on Ultralytics HUB to upload the dataset.", archive)
    return Path(archive)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndjson", required=True, type=Path, help="Path to the NDJSON export.")
    parser.add_argument("--out", type=Path, default=Path("dataset"), help="Output dataset dir.")
    args = parser.parse_args()
    convert(args.ndjson, args.out)


if __name__ == "__main__":
    main()
