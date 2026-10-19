"""Tests for the opt-in `validate=True` checks of the readers."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import iorec

Reader = Callable[..., Any]
Mesh = tuple[np.ndarray, np.ndarray, np.ndarray]

# The plain readers stream from disk; the `*_with_metadata` readers parse an in-memory
# buffer. Both paths must apply the same checks.
READERS_3D = [
    pytest.param(iorec.load_rec, id="load_rec"),
    pytest.param(lambda *a, **k: iorec.load_rec_with_metadata(*a, **k)[:3], id="meta"),
]
READERS_2D = [
    pytest.param(iorec.load_2d_rec, id="load_2d_rec"),
    pytest.param(
        lambda *a, **k: iorec.load_2d_rec_with_metadata(*a, **k)[:3],
        id="meta_2d",
    ),
]


def _tetrahedron() -> Mesh:
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    triangles = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int64)
    labels = np.tile(np.array([[0, 1]], dtype=np.int64), (4, 1))
    return points, triangles, labels


def _square() -> Mesh:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    segments = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    labels = np.tile(np.array([[0, 1]], dtype=np.int64), (4, 1))
    return points, segments, labels


def _binary_3d(points: np.ndarray, triangles: np.ndarray, labels: np.ndarray) -> bytes:
    records = np.empty(
        len(triangles),
        dtype=[("triangles", "<i8", (3,)), ("labels", "<i4", (2,))],
    )
    records["triangles"], records["labels"] = triangles, labels
    return (
        struct.pack("<Q", len(points))
        + np.asarray(points, dtype="<f8").tobytes()
        + struct.pack("<Q", len(triangles))
        + records.tobytes()
    )


def test_rec_format_error_is_a_value_error() -> None:
    """Callers catching ValueError keep working."""
    assert issubclass(iorec.RecFormatError, ValueError)


@pytest.mark.parametrize("reader", READERS_3D)
@pytest.mark.parametrize("asset", ["tests/assets/binary.rec", "tests/assets/text.rec"])
def test_valid_3d_assets_pass_unchanged(reader: Reader, asset: str) -> None:
    """Validation accepts the 3D assets and returns the same arrays as before."""
    checked = reader(asset, validate=True)
    plain = iorec.load_rec(asset)
    for a, b in zip(checked, plain):
        assert a.dtype == b.dtype
        assert np.array_equal(a, b)


@pytest.mark.parametrize("reader", READERS_2D)
@pytest.mark.parametrize(
    "asset",
    ["tests/assets/binary_2d.rec", "tests/assets/text_2d.rec"],
)
def test_valid_2d_assets_pass_unchanged(reader: Reader, asset: str) -> None:
    """Validation accepts the 2D assets and returns the same arrays as before."""
    checked = reader(asset, validate=True)
    plain = iorec.load_2d_rec(asset)
    for a, b in zip(checked, plain):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("reader", READERS_3D)
@pytest.mark.parametrize(
    "asset",
    ["tests/assets/binary_2d.rec", "tests/assets/text_2d.rec"],
)
def test_2d_file_read_as_3d_is_rejected(reader: Reader, asset: str) -> None:
    """A 2D file read by the 3D reader is reported instead of misread."""
    with pytest.raises(iorec.RecFormatError):
        reader(asset, validate=True)


def test_default_behaviour_is_unchanged_on_a_malformed_file(tmp_path: Path) -> None:
    """Without `validate`, a binary file with trailing garbage still reads fine."""
    expected = _tetrahedron()
    filename = tmp_path / "garbage.rec"
    filename.write_bytes(_binary_3d(*expected) + b"trailing garbage")
    with pytest.raises(iorec.RecFormatError):
        iorec.load_rec(filename, validate=True)
    for read, wanted in zip(iorec.load_rec(filename), expected):
        assert np.array_equal(read, wanted)


@pytest.mark.parametrize("reader", READERS_3D)
@pytest.mark.parametrize("binary_mode", [True, False])
def test_appendix_is_accepted(
    reader: Reader,
    tmp_path: Path,
    binary_mode: bool,
) -> None:
    """A trailing metadata appendix is not mistaken for trailing garbage."""
    filename = tmp_path / "mesh.rec"
    points, triangles, labels = _tetrahedron()
    iorec.save_rec(
        filename,
        points,
        triangles,
        labels,
        binary_mode=binary_mode,
        metadata={"coordinate_frame": "voxel_index", "spacing": (1.0, 0.5, 0.5)},
    )
    out = reader(filename, validate=True)
    assert np.array_equal(out[0], points)
    assert np.array_equal(out[1], triangles)


@pytest.mark.parametrize("reader", READERS_2D)
@pytest.mark.parametrize("binary_mode", [True, False])
def test_appendix_is_accepted_2d(
    reader: Reader,
    tmp_path: Path,
    binary_mode: bool,
) -> None:
    """A trailing metadata appendix is accepted by the 2D readers."""
    filename = tmp_path / "mesh.rec"
    iorec.save_2d_rec(
        filename,
        *_square(),
        binary_mode=binary_mode,
        metadata={"coordinate_frame": "physical"},
    )
    out = reader(filename, validate=True)
    assert np.array_equal(out[1], _square()[1])


@pytest.mark.parametrize("reader", READERS_3D)
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda b: b[:4], "point count"),
        (lambda b: b[:40], "points"),
        (lambda b: b[:-1], "triangles"),
        (lambda b: b + b"\x00", "trailing bytes"),
        (lambda b: b + b"# not an appendix\n", "trailing bytes"),
    ],
    ids=["no-header", "short-points", "short-triangles", "extra-byte", "extra-text"],
)
def test_binary_size_is_checked(
    reader: Reader,
    tmp_path: Path,
    mutate: Callable[[bytes], bytes],
    message: str,
) -> None:
    """The binary size must match the declared counts (plus an optional appendix)."""
    filename = tmp_path / "mesh.rec"
    filename.write_bytes(mutate(_binary_3d(*_tetrahedron())))
    with pytest.raises(iorec.RecFormatError, match=message):
        reader(filename, validate=True)


def test_huge_declared_count_is_rejected_before_reading(tmp_path: Path) -> None:
    """A corrupt header is reported without trying to allocate its count."""
    filename = tmp_path / "mesh.rec"
    filename.write_bytes(struct.pack("<Q", 2**60) + b"\x00" * 64)
    with pytest.raises(iorec.RecFormatError, match="truncated binary points"):
        iorec.load_rec(filename, validate=True)


TEXT_POINTS = "4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
TEXT_OK = TEXT_POINTS + "4\n0 1 2 0 1\n0 3 1 0 1\n0 2 3 0 1\n1 3 2 0 1\n"
TEXT_FOUR_COLUMNS = TEXT_POINTS + "4\n0 1 2 0\n0 3 1 0\n0 2 3 0\n1 3 2 0\n"
TEXT_SIX_COLUMNS = (
    TEXT_POINTS + "4\n0 1 2 0 1 9\n0 3 1 0 1 9\n0 2 3 0 1 9\n1 3 2 0 1 9\n"
)


@pytest.mark.parametrize("reader", READERS_3D)
@pytest.mark.parametrize(
    ("text", "message"),
    [
        (TEXT_OK.replace("4\n0 1 2", "5\n0 1 2"), "shape"),
        (TEXT_OK.replace("0 0 1\n", "0 0 1 7\n"), "point block"),
        (TEXT_OK.replace("1 3 2 0 1", "1 3 2 0 1 5"), "triangle block"),
        (TEXT_FOUR_COLUMNS, "columns, expected 5"),
        (TEXT_OK.replace("0 3 1", "0 3 x"), "triangle block"),
        (TEXT_OK.replace("4\n0 0 0", "-4\n0 0 0"), "non-negative"),
        (TEXT_OK.replace("4\n0 0 0", "four\n0 0 0"), "invalid point count"),
        (TEXT_OK + "1 2 3\n", "unexpected content"),
        (TEXT_OK.split("4\n0 1 2")[0], "invalid triangle count"),
    ],
    ids=[
        "missing-row",
        "ragged-points",
        "ragged-triangles",
        "four-columns",
        "non-integer",
        "negative-count",
        "bad-count",
        "trailing-row",
        "no-triangle-block",
    ],
)
def test_text_structure_is_checked(
    reader: Reader,
    tmp_path: Path,
    text: str,
    message: str,
) -> None:
    """Text files must have the declared rows and columns, and nothing else."""
    filename = tmp_path / "mesh.rec"
    filename.write_text(text)
    with pytest.raises(iorec.RecFormatError, match=message):
        reader(filename, validate=True)


def test_six_column_triangles_are_rejected(tmp_path: Path) -> None:
    """Six columns would silently drop one value without the column check."""
    filename = tmp_path / "mesh.rec"
    filename.write_text(TEXT_SIX_COLUMNS)
    with pytest.raises(iorec.RecFormatError, match="columns, expected 5"):
        iorec.load_rec(filename, validate=True)


def _bad_index(p: np.ndarray, t: np.ndarray, lab: np.ndarray) -> Mesh:
    t = t.copy()
    t[2, 1] = 4
    return p, t, lab


def _negative_index(p: np.ndarray, t: np.ndarray, lab: np.ndarray) -> Mesh:
    t = t.copy()
    t[0, 0] = -1
    return p, t, lab


def _nan(p: np.ndarray, t: np.ndarray, lab: np.ndarray) -> Mesh:
    p = p.copy()
    p[3, 2] = np.nan
    return p, t, lab


def _inf(p: np.ndarray, t: np.ndarray, lab: np.ndarray) -> Mesh:
    p = p.copy()
    p[1, 0] = -np.inf
    return p, t, lab


def _negative_label(p: np.ndarray, t: np.ndarray, lab: np.ndarray) -> Mesh:
    lab = lab.copy()
    lab[1] = [0, -1]
    return p, t, lab


def _same_label(p: np.ndarray, t: np.ndarray, lab: np.ndarray) -> Mesh:
    lab = lab.copy()
    lab[3] = [2, 2]
    return p, t, lab


@pytest.mark.parametrize("reader", READERS_3D)
@pytest.mark.parametrize("binary_mode", [True, False])
@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_bad_index, r"triangle 2 has a vertex index outside \[0, 4\)"),
        (_negative_index, "triangle 0 has a vertex index outside"),
        (_nan, "point 3 has a non-finite coordinate"),
        (_inf, "point 1 has a non-finite coordinate"),
        (_negative_label, "triangle 1 has a negative label"),
        (_same_label, "triangle 3 separates label 2 from itself"),
    ],
    ids=["index", "negative-index", "nan", "inf", "negative-label", "same-labels"],
)
def test_content_is_checked(
    reader: Reader,
    tmp_path: Path,
    binary_mode: bool,
    corrupt: Callable[..., Mesh],
    message: str,
) -> None:
    """Coordinates, indices and labels are checked in both modes."""
    filename = tmp_path / "mesh.rec"
    iorec.save_rec(filename, *corrupt(*_tetrahedron()), binary_mode=binary_mode)
    with pytest.raises(iorec.RecFormatError, match=message) as exc_info:
        reader(filename, validate=True)
    assert str(filename.resolve()) in str(exc_info.value)
    reader(filename)  # without validate the file still reads


@pytest.mark.parametrize("reader", READERS_2D)
@pytest.mark.parametrize("binary_mode", [True, False])
def test_2d_content_is_checked(
    reader: Reader,
    tmp_path: Path,
    binary_mode: bool,
) -> None:
    """The 2D readers apply the same checks to segments."""
    filename = tmp_path / "mesh.rec"
    points, segments, labels = _square()
    segments = segments.copy()
    segments[3, 1] = 7
    iorec.save_2d_rec(filename, points, segments, labels, binary_mode=binary_mode)
    with pytest.raises(iorec.RecFormatError, match="segment 3 has a vertex index"):
        reader(filename, validate=True)


@pytest.mark.parametrize("reader", READERS_2D)
def test_2d_binary_size_is_checked(reader: Reader, tmp_path: Path) -> None:
    """A truncated 2D binary file is reported."""
    filename = tmp_path / "mesh.rec"
    iorec.save_2d_rec(filename, *_square(), binary_mode=True)
    filename.write_bytes(filename.read_bytes()[:-3])
    with pytest.raises(iorec.RecFormatError, match="truncated binary segments"):
        reader(filename, validate=True)


@pytest.mark.parametrize("reader", READERS_3D)
@pytest.mark.parametrize("binary_mode", [True, False])
def test_empty_mesh_is_valid(reader: Reader, tmp_path: Path, binary_mode: bool) -> None:
    """A mesh with no points and no triangles is well formed."""
    filename = tmp_path / "empty.rec"
    iorec.save_rec(
        filename,
        np.empty((0, 3)),
        np.empty((0, 3), dtype=np.int64),
        np.empty((0, 2), dtype=np.int64),
        binary_mode=binary_mode,
    )
    points, triangles, labels = reader(filename, validate=True)
    assert points.shape == (0, 3)
    assert triangles.shape == (0, 3)
    assert labels.shape == (0, 2)
