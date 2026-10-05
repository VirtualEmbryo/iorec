"""The vectorized text writer produces the same text as the original row loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import iorec


def _loop_writer_3d(
    points: np.ndarray,
    triangles: np.ndarray,
    labels: np.ndarray,
) -> str:
    """The text writer of iorec 1.0.9 (before vectorization), kept as the reference."""
    content = str(len(points)) + "\n"
    for i in range(len(points)):
        content += f"{points[i][0]} {points[i][1]} {points[i][2]}\n"
    content += str(len(triangles)) + "\n"
    for i in range(len(triangles)):
        content += f"{triangles[i][0]} {triangles[i][1]} {triangles[i][2]} "
        content += f"{labels[i][0]} {labels[i][1]}\n"
    return content


def _loop_writer_2d(
    points: np.ndarray,
    segments: np.ndarray,
    labels: np.ndarray,
) -> str:
    content = str(len(points)) + "\n"
    for i in range(len(points)):
        content += f"{points[i][0]} {points[i][1]}\n"
    content += str(len(segments)) + "\n"
    for i in range(len(segments)):
        content += f"{segments[i][0]} {segments[i][1]} "
        content += f"{labels[i][0]} {labels[i][1]}\n"
    return content


def _hard_points(n: int, dtype: str) -> np.ndarray:
    """Coordinates over the whole float range, with the special values."""
    rng = np.random.default_rng(0)
    exponents = rng.integers(-300, 300, size=n * 3)
    values = rng.standard_normal(n * 3) * 10.0 ** exponents.astype(np.float64)
    special = [0.0, -0.0, np.nan, np.inf, -np.inf, 5e-324, 1e16, 1e15, 1e-5, 0.1]
    values[: len(special)] = special
    with np.errstate(over="ignore", invalid="ignore"):  # casts to float16 or int
        return values.reshape(n, 3).astype(dtype)


@pytest.mark.parametrize("asset", ["tests/assets/text.rec", "tests/assets/binary.rec"])
def test_text_writer_matches_loop_writer_on_assets(tmp_path: Path, asset: str) -> None:
    """Same text as the original writer on the 3D assets."""
    points, triangles, labels = iorec.load_rec(asset)
    filename = tmp_path / "out.rec"
    iorec.save_rec(filename, points, triangles, labels)
    assert filename.read_text() == _loop_writer_3d(points, triangles, labels)


@pytest.mark.parametrize(
    "asset",
    ["tests/assets/text_2d.rec", "tests/assets/binary_2d.rec"],
)
def test_2d_text_writer_matches_loop_writer_on_assets(
    tmp_path: Path,
    asset: str,
) -> None:
    """Same text as the original writer on the 2D assets."""
    points, segments, labels = iorec.load_2d_rec(asset)
    filename = tmp_path / "out.rec"
    iorec.save_2d_rec(filename, points, segments, labels)
    assert filename.read_text() == _loop_writer_2d(points, segments, labels)


@pytest.mark.parametrize("point_dtype", ["<f8", ">f8", "<f4", "<f2", "<i8"])
@pytest.mark.parametrize("index_dtype", ["<i8", "<u8", "<i4", ">i8"])
def test_text_writer_matches_loop_writer_on_any_dtype(
    tmp_path: Path,
    point_dtype: str,
    index_dtype: str,
) -> None:
    """Same text for every input dtype, including extreme and special values."""
    n = 200
    points = _hard_points(n, point_dtype)
    rng = np.random.default_rng(1)
    triangles = rng.integers(0, n, size=(2 * n, 3)).astype(index_dtype)
    labels = rng.integers(0, 50, size=(2 * n, 2)).astype(index_dtype)
    filename = tmp_path / "out.rec"
    iorec.save_rec(filename, points, triangles, labels)
    assert filename.read_text() == _loop_writer_3d(points, triangles, labels)


def test_text_round_trip_is_exact_for_float64(tmp_path: Path) -> None:
    """Text mode keeps every float64 bit (shortest round-trip repr)."""
    points = _hard_points(500, "<f8")
    triangles = np.zeros((1, 3), dtype=np.int64)
    labels = np.array([[0, 1]], dtype=np.int64)
    filename = tmp_path / "out.rec"
    iorec.save_rec(filename, points, triangles, labels)
    back, _, _ = iorec.load_rec(filename)
    finite = np.isfinite(points)
    assert np.array_equal(back[finite].view(np.uint64), points[finite].view(np.uint64))
    assert np.array_equal(np.isnan(back), np.isnan(points))
    assert np.array_equal(back[np.isinf(points)], points[np.isinf(points)])
