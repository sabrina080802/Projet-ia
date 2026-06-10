"""Unit tests for the shared class helpers."""

from sharp import classes


def test_class_names_count():
    assert len(classes.CLASS_NAMES) == 6


def test_fingers_for_class_matches_index():
    for class_id in range(6):
        assert classes.fingers_for_class(class_id) == class_id


def test_names_mapping_is_indexed():
    mapping = classes.names_mapping()
    assert mapping[0] == "0_doigt"
    assert mapping[5] == "5_doigts"
