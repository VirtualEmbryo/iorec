from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rec_loader


def test_binary_file():
    points, triangles, labels = rec_loader.open_rec("tests/assets/binary.rec")

    assert points.dtype == np.float64
    assert triangles.dtype == np.ulonglong
    assert labels.dtype == np.uint

    assert points.shape == (8137, 3)
    assert triangles.shape == (17000, 3)
    assert labels.shape == (17000, 2)

    assert np.max(triangles) == 8136  # all points used in this examples
    assert np.max(labels) == 16  # 16 cells + exterior


def test_text_file():
    points, triangles, labels = rec_loader.open_rec("tests/assets/text.rec")

    assert points.dtype == np.float64
    assert triangles.dtype == np.ulonglong
    assert labels.dtype == np.uint

    assert points.shape == (142, 3)
    assert triangles.shape == (318, 3)
    assert labels.shape == (318, 2)

    assert np.max(triangles) == 141  # all points used in this examples
    assert np.max(labels) == 4  # 4 cells + exterior


def test_non_existing_file():
    with pytest.raises(FileNotFoundError) as exc_info:
        rec_loader.open_rec("tests/assets/NOT_A_FILE.rec")

    assert (
        exc_info.value.args[0]
        == f"{Path('tests/assets/NOT_A_FILE.rec').resolve()} not found."
    )


def test_non_rec_file():
    with pytest.raises(ValueError) as exc_info:
        rec_loader.open_rec("tests/assets/README.md")

    assert (
        exc_info.value.args[0]
        == f"{Path('tests/assets/README.md').resolve()} is not a .rec or .arec file."
    )
