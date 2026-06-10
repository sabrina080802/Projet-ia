"""Unit tests for the deterministic split logic."""

from pathlib import Path

from sharp.local import preparation


def _make_pairs(n: int) -> list[tuple[Path, Path]]:
    return [(Path(f"img_{i}.jpg"), Path(f"img_{i}.txt")) for i in range(n)]


def test_split_ratios_sum_to_total():
    splits = preparation._split_pairs(_make_pairs(100))
    assert len(splits["train"]) == 60
    assert len(splits["val"]) == 20
    assert len(splits["test"]) == 20


def test_split_is_deterministic_with_seed():
    first = preparation._split_pairs(_make_pairs(50))
    second = preparation._split_pairs(_make_pairs(50))
    assert first["train"] == second["train"]
    assert first["test"] == second["test"]


def test_no_pair_appears_in_two_splits():
    splits = preparation._split_pairs(_make_pairs(100))
    seen = splits["train"] + splits["val"] + splits["test"]
    assert len(seen) == len(set(seen)) == 100


def test_existing_split_is_reused_from_paths():
    pairs = [
        (Path("images/train/a.jpg"), Path("labels/train/a.txt")),
        (Path("images/valid/b.jpg"), Path("labels/valid/b.txt")),
        (Path("images/test/c.jpg"), Path("labels/test/c.txt")),
    ]
    splits = preparation._existing_split(pairs)
    assert splits is not None
    assert [p[0].name for p in splits["train"]] == ["a.jpg"]
    assert [p[0].name for p in splits["val"]] == ["b.jpg"]  # "valid" alias -> val
    assert [p[0].name for p in splits["test"]] == ["c.jpg"]


def test_no_existing_split_falls_back_to_random():
    # Flat layout with no train/val/test segment -> None (caller re-splits).
    assert preparation._existing_split(_make_pairs(10)) is None
