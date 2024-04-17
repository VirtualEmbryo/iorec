"""Test file for the package."""

from __future__ import annotations

from pathlib import Path

import iorec
import numpy as np
import pytest


def test_load_binary_file() -> None:
    """One can load a binary rec mesh file."""
    points, triangles, labels = iorec.load_rec("tests/assets/binary.rec")

    assert points.dtype == np.float64
    assert triangles.dtype == np.int64
    assert labels.dtype == np.int64

    assert points.shape == (8137, 3)
    assert triangles.shape == (17000, 3)
    assert labels.shape == (17000, 2)

    assert np.max(triangles) == 8136  # all points used in this examples
    assert np.max(labels) == 16  # 16 cells + exterior


def test_text_file() -> None:
    """One can load a text rec mesh file."""
    points, triangles, labels = iorec.load_rec("tests/assets/text.rec")

    assert points.dtype == np.float64
    assert triangles.dtype == np.int64
    assert labels.dtype == np.int64

    assert points.shape == (142, 3)
    assert triangles.shape == (318, 3)
    assert labels.shape == (318, 2)

    assert np.max(triangles) == 141  # all points used in this examples
    assert np.max(labels) == 4  # 4 cells + exterior


def test_load_non_existing_file() -> None:
    """Loading a non existing file raises a FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        iorec.load_rec("tests/assets/NOT_A_FILE.rec")

    assert (
        exc_info.value.args[0]
        == f"{Path('tests/assets/NOT_A_FILE.rec').resolve()} not found."
    )


def test_load_non_rec_file() -> None:
    """Load a non-rec file raises a ValueError."""
    with pytest.raises(ValueError, match="not a .rec or .arec file") as exc_info:
        iorec.load_rec("tests/assets/README.md")

    assert (
        exc_info.value.args[0]
        == f"{Path('tests/assets/README.md').resolve()} is not a .rec or .arec file."
    )


def test_load_save_load_binary(tmp_path: Path) -> None:
    """One can save numpy arrays of points, triangles and labels into a .rec file."""
    filename = tmp_path / "bin.rec"
    points, triangles, labels = iorec.load_rec("tests/assets/binary.rec")

    iorec.save_rec(filename, points, triangles, labels, binary_mode=True)

    saved_points, saved_triangles, saved_labels = iorec.load_rec(filename)

    assert np.array_equal(points, saved_points)
    assert np.array_equal(triangles, saved_triangles)
    assert np.array_equal(labels, saved_labels)


def test_load_save_load_text(tmp_path: Path) -> None:
    """One can save numpy arrays of points, triangles and labels into a .rec file."""
    filename = tmp_path / "txt.rec"
    points, triangles, labels = iorec.load_rec("tests/assets/text.rec")

    iorec.save_rec(filename, points, triangles, labels, binary_mode=False)

    saved_points, saved_triangles, saved_labels = iorec.load_rec(filename)

    assert np.array_equal(points, saved_points)
    assert np.array_equal(triangles, saved_triangles)
    assert np.array_equal(labels, saved_labels)


def test_load_save_load_2d_text(tmp_path: Path) -> None:
    """One can save numpy arrays of points, segments and labels into a 2D .rec file."""
    filename = tmp_path / "txt2d.rec"
    points, segments, labels = iorec.load_2d_rec("tests/assets/text_2d.rec")

    iorec.save_2d_rec(filename, points, segments, labels, binary_mode=False)

    saved_points, saved_segments, saved_labels = iorec.load_2d_rec(filename)

    assert np.array_equal(points, saved_points)
    assert np.array_equal(segments, saved_segments)
    assert np.array_equal(labels, saved_labels)


def test_load_save_load_2d_binary(tmp_path: Path) -> None:
    """One can save numpy arrays of points, segments and labels into a 2D .rec file."""
    filename = tmp_path / "bin2d.rec"
    points, segments, labels = iorec.load_2d_rec("tests/assets/binary_2d.rec")

    iorec.save_2d_rec(filename, points, segments, labels, binary_mode=True)

    print("SAVED DONE")

    saved_points, saved_segments, saved_labels = iorec.load_2d_rec(filename)

    assert np.array_equal(points, saved_points)
    assert np.array_equal(segments, saved_segments)
    assert np.array_equal(labels, saved_labels)
