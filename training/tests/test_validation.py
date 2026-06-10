"""Unit tests for the validation stage label checks."""

from pathlib import Path

from sharp.local import validation


def _write(tmp_path: Path, content: str) -> Path:
    label = tmp_path / "img.txt"
    label.write_text(content)
    return label


def test_valid_label_has_no_problems(tmp_path):
    label = _write(tmp_path, "2 0.5 0.5 0.2 0.3\n")
    assert validation._check_label(label, num_classes=6) == []


def test_negative_coordinate_flagged(tmp_path):
    label = _write(tmp_path, "1 -0.1 0.5 0.2 0.3\n")
    problems = validation._check_label(label, num_classes=6)
    assert any("negative" in p for p in problems)


def test_class_out_of_range_flagged(tmp_path):
    label = _write(tmp_path, "9 0.5 0.5 0.2 0.3\n")
    problems = validation._check_label(label, num_classes=6)
    assert any("out of range" in p for p in problems)


def test_wrong_field_count_flagged(tmp_path):
    label = _write(tmp_path, "1 0.5 0.5\n")
    problems = validation._check_label(label, num_classes=6)
    assert any("expected 5 values" in p for p in problems)


def test_unnormalized_coordinate_flagged(tmp_path):
    label = _write(tmp_path, "1 0.5 0.5 1.4 0.3\n")
    problems = validation._check_label(label, num_classes=6)
    assert any("not normalized" in p for p in problems)


def test_box_extending_outside_image_flagged(tmp_path):
    # Every value is in [0, 1] but cx + w/2 = 1.15 spills past the right edge.
    label = _write(tmp_path, "2 0.95 0.5 0.4 0.3\n")
    problems = validation._check_label(label, num_classes=6)
    assert any("outside the image" in p for p in problems)


def test_zero_area_box_flagged(tmp_path):
    label = _write(tmp_path, "2 0.5 0.5 0.0 0.3\n")
    problems = validation._check_label(label, num_classes=6)
    assert any("zero-area" in p for p in problems)


def test_check_image_rejects_truncated_file(tmp_path):
    from PIL import Image

    good = tmp_path / "good.jpg"
    Image.new("RGB", (64, 64), (128, 128, 128)).save(good, "JPEG")
    assert validation._check_image(good) is True

    truncated = tmp_path / "bad.jpg"
    truncated.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    assert validation._check_image(truncated) is False
