"""Canonical class definitions shared by both pipelines.

The six classes are fixed by the project statement and must not change. Each
class name encodes the number of raised fingers, which the serving dashboard
sums across every detected hand.
"""

from __future__ import annotations

from config import settings

# Ordered class names; the list index is the YOLO class id.
CLASS_NAMES: list[str] = list(settings.data.class_names)


def fingers_for_class(class_id: int) -> int:
    """Return the finger count encoded by a YOLO class id.

    Args:
        class_id: Zero-based class index as produced by the model.

    Returns:
        The number of fingers for that class (``0_doigt`` -> 0, ``5_doigts`` -> 5).

    Raises:
        IndexError: If ``class_id`` is outside the known class range.
    """
    name = CLASS_NAMES[class_id]
    return int(name.split("_", 1)[0])


def names_mapping() -> dict[int, str]:
    """Return the ``{id: name}`` mapping expected in an Ultralytics ``data.yaml``."""
    return dict(enumerate(CLASS_NAMES))
