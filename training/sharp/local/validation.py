"""Stage 2 - Validation.

Verify the extracted dataset before preparing it: images must open without
errors and every YOLO label must be structurally and numerically coherent
(valid class id, no negative coordinates, normalized values in ``[0, 1]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from sharp.classes import CLASS_NAMES
from sharp.logging_utils import get_logger

logger = get_logger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ValidationReport:
    """Outcome of a dataset validation pass."""

    images_checked: int = 0
    labels_checked: int = 0
    corrupt_images: list[str] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no corrupt image and no invalid label were found."""
        return not self.corrupt_images and not self.invalid_labels


def _iter_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]


def _label_for_image(image: Path) -> Path:
    """Map ``.../images/.../foo.jpg`` to ``.../labels/.../foo.txt``."""
    parts = [("labels" if part == "images" else part) for part in image.parts]
    return Path(*parts).with_suffix(".txt")


def _check_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def _check_label(path: Path, num_classes: int) -> list[str]:
    """Return a list of human-readable problems for one label file."""
    problems: list[str] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            problems.append(f"{path}:{lineno} expected 5 values, got {len(fields)}")
            continue
        try:
            class_id = int(float(fields[0]))
            cx, cy, w, h = (float(v) for v in fields[1:])
        except ValueError:
            problems.append(f"{path}:{lineno} non-numeric value")
            continue
        if not 0 <= class_id < num_classes:
            problems.append(f"{path}:{lineno} class id {class_id} out of range")
        if any(v < 0 for v in (cx, cy, w, h)):
            problems.append(f"{path}:{lineno} negative coordinate")
        if any(v > 1 for v in (cx, cy, w, h)):
            problems.append(f"{path}:{lineno} coordinate > 1 (not normalized)")
    return problems


def validate(root: str | Path) -> ValidationReport:
    """Validate every image and matching label under ``root``.

    Args:
        root: Directory containing the extracted dataset (images and labels).

    Returns:
        A :class:`ValidationReport` summarizing the findings.
    """
    root = Path(root)
    report = ValidationReport()

    for image in _iter_images(root):
        report.images_checked += 1
        if not _check_image(image):
            report.corrupt_images.append(str(image))

        label = _label_for_image(image)
        if label.exists():
            report.labels_checked += 1
            report.invalid_labels.extend(_check_label(label, len(CLASS_NAMES)))

    logger.info(
        "Validation: %d images, %d labels, %d corrupt, %d invalid",
        report.images_checked,
        report.labels_checked,
        len(report.corrupt_images),
        len(report.invalid_labels),
    )
    return report
