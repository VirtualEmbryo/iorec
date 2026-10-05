"""Binary files are little-endian on every machine, and written from any input dtype."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import iorec


def _expected_3d_bytes(
    points: np.ndarray,
    triangles: np.ndarray,
    labels: np.ndarray,
) -> bytes:
    """The documented layout, built value by value with explicit '<' formats."""
    out = struct.pack("<Q", len(points))
    out += b"".join(struct.pack("<3d", *p) for p in points.tolist())
    out += struct.pack("<Q", len(triangles))
    for t, lab in zip(triangles.tolist(), labels.tolist()):
        out += struct.pack("<3q2i", *t, *lab)
    return out


def _expected_2d_bytes(
    points: np.ndarray,
    segments: np.ndarray,
    labels: np.ndarray,
) -> bytes:
    out = struct.pack("<Q", len(points))
    out += b"".join(struct.pack("<2d", *p) for p in points.tolist())
    out += struct.pack("<Q", len(segments))
    for s, lab in zip(segments.tolist(), labels.tolist()):
        out += struct.pack("<2q2i", *s, *lab)
    return out


def test_binary_writer_matches_little_endian_layout(tmp_path: Path) -> None:
    """save_rec writes the documented little-endian layout (3D)."""
    points, triangles, labels = iorec.load_rec("tests/assets/text.rec")
    filename = tmp_path / "bin.rec"
    iorec.save_rec(filename, points, triangles, labels, binary_mode=True)
    assert filename.read_bytes() == _expected_3d_bytes(points, triangles, labels)


def test_2d_binary_writer_matches_little_endian_layout(tmp_path: Path) -> None:
    """save_2d_rec writes the documented little-endian layout (2D)."""
    points, segments, labels = iorec.load_2d_rec("tests/assets/text_2d.rec")
    filename = tmp_path / "bin2d.rec"
    iorec.save_2d_rec(filename, points, segments, labels, binary_mode=True)
    assert filename.read_bytes() == _expected_2d_bytes(points, segments, labels)


@pytest.mark.parametrize(
    "asset",
    ["tests/assets/binary.rec", "tests/assets/binary_2d.rec"],
)
def test_binary_round_trip_is_byte_identical(tmp_path: Path, asset: str) -> None:
    """Reading then writing a binary asset reproduces the file byte for byte."""
    is_2d = asset.endswith("_2d.rec")
    load = iorec.load_2d_rec if is_2d else iorec.load_rec
    save = iorec.save_2d_rec if is_2d else iorec.save_rec
    filename = tmp_path / "copy.rec"
    save(filename, *load(asset), binary_mode=True)
    assert filename.read_bytes() == Path(asset).read_bytes()


@pytest.mark.parametrize(
    ("point_dtype", "index_dtype", "label_dtype"),
    [
        (">f8", ">i8", ">i8"),  # big-endian arrays, e.g. from a FITS or MATLAB file
        ("<f4", "<i4", "<u2"),  # narrower types
        ("<f8", "<u8", "<i4"),
    ],
)
@pytest.mark.parametrize("order", ["C", "F"])
def test_binary_output_does_not_depend_on_input_dtype(
    tmp_path: Path,
    point_dtype: str,
    index_dtype: str,
    label_dtype: str,
    order: str,
) -> None:
    """Any input dtype, byte order or memory order gives the same file.

    Before explicit little-endian writing, float32 points were written as raw float32
    bytes into a float64 layout (an unreadable file) and big-endian arrays were written
    byte-swapped.
    """
    points, triangles, labels = iorec.load_rec("tests/assets/text.rec")
    points = np.round(points, 3)  # exactly representable change for float32 below
    points = points.astype(np.float32).astype(np.float64)
    reference = tmp_path / "reference.rec"
    iorec.save_rec(reference, points, triangles, labels, binary_mode=True)

    converted = tmp_path / "converted.rec"
    iorec.save_rec(
        converted,
        np.asarray(points, dtype=point_dtype, order=order),
        np.asarray(triangles, dtype=index_dtype, order=order),
        np.asarray(labels, dtype=label_dtype, order=order),
        binary_mode=True,
    )
    assert converted.read_bytes() == reference.read_bytes()


@pytest.mark.parametrize("reader", [iorec.load_rec, iorec.load_rec_with_metadata])
def test_binary_reader_returns_native_arrays(reader: object) -> None:
    """Returned arrays are native-endian float64/int64, as documented."""
    arrays = reader("tests/assets/binary.rec")  # type: ignore[operator]
    for array, dtype in zip(arrays[:3], (np.float64, np.int64, np.int64)):
        assert array.dtype == dtype
        assert array.dtype.isnative
        assert array.flags.c_contiguous
        assert array.flags.writeable
