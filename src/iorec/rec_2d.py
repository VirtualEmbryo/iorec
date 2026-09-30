"""IO for 2D .rec meshes."""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from iorec._utils import (
    _is_binary,
    _is_binary_bytes,
    _resolve_rec_path,
    _write_geometry_with_appendix,
)
from iorec.rec_appendix import _split_rec_bytes


def load_2d_rec(
    filename: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a 2D rec mesh file.

    Args:
        filename (str | Path): The path to the file on disk.

    Raises:
        ValueError: If the path given is not a .rec or .arec file.
        FileNotFoundError: If the file doesn't exist.

    Returns:
        tuple[np.array, np.array, np.array]: (points, segments, labels), with:
        - points: np.array of shape (num_points, 2) (float64)
        - segments: np.array of shape (num_segments, 2) (ulonglong)
        - labels: np.array of shape (num_segments, 2) (uint)
    """
    filename = _resolve_rec_path(filename)
    return _load_2d_rec_file(filename)


def load_2d_rec_with_metadata(
    filename: str | Path,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    dict[str, Any] | None,
]:
    """Read a 2D rec mesh file together with its metadata appendix, if any.

    Reads the file once and splits geometry from metadata in memory -- unlike calling
    `load_2d_rec` and `load_rec_appendix` separately, which would each read the whole
    file on their own.

    Args:
        filename (str | Path): The path to the file on disk.

    Raises:
        ValueError: If the path given is not a .rec or .arec file.
        FileNotFoundError: If the file doesn't exist.

    Returns:
        tuple[np.array, np.array, np.array, dict | None]: (points, segments, labels,
        metadata), with points/segments/labels as in `load_2d_rec`, and metadata as in
        `load_rec_appendix` (`None` if the file has no appendix).
    """
    filename = _resolve_rec_path(filename)
    geometry_bytes, metadata = _split_rec_bytes(filename.read_bytes())

    if _is_binary_bytes(geometry_bytes):
        points, segments, labels = _parse_2d_binary_rec_bytes(geometry_bytes)
    else:
        points, segments, labels = _parse_2d_text_rec_bytes(geometry_bytes)

    return points, segments, labels, metadata


def _load_2d_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    if _is_binary(filename):
        return _load_2d_binary_rec_file(filename)
    else:
        return _load_2d_text_rec_file(filename)


def _load_2d_binary_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    with Path.open(filename, "rb") as f:
        # First we read the number of vertices
        (number_of_points,) = struct.unpack("Q", f.read(8))
        # Then we read the vertices : they are in a 2D space
        points = np.fromfile(f, count=2 * number_of_points, dtype=np.float64).reshape(
            (number_of_points, 2),
        )

        # Number of segments
        (number_of_segments,) = struct.unpack("Q", f.read(8))
        # Each line defines a segment and 2 regions labels
        dtype_segmens_and_labels = np.dtype(
            [("segments", np.int64, (2,)), ("labels", np.int32, (2,))],
        )
        segments_and_labels = np.fromfile(
            f,
            count=number_of_segments,
            dtype=dtype_segmens_and_labels,
        )

    return (
        points,
        segments_and_labels["segments"].astype(np.int64),
        segments_and_labels["labels"].astype(np.int64),
    )


def _load_2d_text_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    with Path.open(filename) as f:
        # First we read the number of vertices
        number_of_points = int(f.readline().strip())
        # Then we read the vertices : they are in a 2D space
        points = (
            np.loadtxt(f, max_rows=number_of_points, dtype=np.float64, ndmin=2)
            if number_of_points > 0
            else np.empty((0, 2), dtype=np.float64)
        )

        # Number of segments
        number_of_segments = int(f.readline().strip())
        # Each line defines a segment and 2 regions labels
        segments_and_labels = (
            np.loadtxt(f, max_rows=number_of_segments, dtype=np.int64, ndmin=2)
            if number_of_segments > 0
            else np.empty((0, 4), dtype=np.int64)
        )

    return (
        points,
        segments_and_labels[:, 0:2],
        segments_and_labels[:, 2:4],
    )


def _parse_2d_binary_rec_bytes(
    data: bytes,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Parse geometry from an in-memory buffer.

    E.g. the geometry half of a buffer already split by
    `iorec.rec_appendix._split_rec_bytes`.

    Kept separate from `_load_2d_binary_rec_file` (which streams from disk with
    `np.fromfile`) rather than unifying the two: unifying would make plain
    `load_2d_rec` hold the whole file in memory as `bytes` *and* as the parsed array at
    once, which for very large files is a real peak-memory regression that
    `load_2d_rec` shouldn't pay just so `load_2d_rec_with_metadata` can avoid a second
    read.
    """
    (number_of_points,) = struct.unpack_from("Q", data)
    offset = 8
    points_count = 2 * number_of_points
    points = (
        np.frombuffer(data, dtype=np.float64, count=points_count, offset=offset)
        .reshape((number_of_points, 2))
        .copy()
    )
    offset += points_count * 8

    (number_of_segments,) = struct.unpack_from("Q", data, offset)
    offset += 8
    dtype_segments_and_labels = np.dtype(
        [("segments", np.int64, (2,)), ("labels", np.int32, (2,))],
    )
    segments_and_labels = np.frombuffer(
        data,
        dtype=dtype_segments_and_labels,
        count=number_of_segments,
        offset=offset,
    )

    return (
        points,
        segments_and_labels["segments"].astype(np.int64),
        segments_and_labels["labels"].astype(np.int64),
    )


def _parse_2d_text_rec_bytes(
    data: bytes,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Parse geometry from an in-memory buffer.

    See `_parse_2d_binary_rec_bytes` for why this is kept separate from
    `_load_2d_text_rec_file` rather than unifying the two.
    """
    with io.StringIO(data.decode()) as f:
        number_of_points = int(f.readline().strip())
        points = (
            np.loadtxt(f, max_rows=number_of_points, dtype=np.float64, ndmin=2)
            if number_of_points > 0
            else np.empty((0, 2), dtype=np.float64)
        )

        number_of_segments = int(f.readline().strip())
        segments_and_labels = (
            np.loadtxt(f, max_rows=number_of_segments, dtype=np.int64, ndmin=2)
            if number_of_segments > 0
            else np.empty((0, 4), dtype=np.int64)
        )

    return (
        points,
        segments_and_labels[:, 0:2],
        segments_and_labels[:, 2:4],
    )


def save_2d_rec(
    filename: str | Path,
    points: NDArray[np.float64],
    segments: NDArray[np.int64],
    labels: NDArray[np.int64],
    binary_mode: bool = False,
    *,
    metadata: dict[str, Any] | None = None,
    written_by: str | None = None,
) -> None:
    """Save a 2D rec mesh file at filename. Binary or text mode are available.

    Binary mode is more compact but might not be universally understood.

    Args:
        filename (str | Path): where to save the mesh file.
        points (NDArray[np.float64]): Mesh points.
        segments (NDArray[np.int64]): Mesh segments (topology).
        labels (NDArray[np.int64]): Segments multimaterial labels.
        binary_mode (bool, optional): Choose binary mode. Defaults to False.
        metadata (dict[str, Any] | None, optional): appendix fields, e.g.
            `coordinate_frame`, `axis_order`, `spacing`, `spacing_unit`,
            `source_shape`, `crop_offset`. Defaults to `None`, which writes nothing
            extra -- byte-for-byte the same file as before this option existed. When
            supplied, a trailing metadata appendix is appended after the geometry data;
            see `iorec.rec_appendix.write_rec_appendix`.
        written_by (str | None, optional): a version string identifying the program
            that produced the mesh, stamped on the appendix's `written_by` line.
            Defaults to `None`, which stamps iorec's own version there instead --
            unchanged from before this parameter existed. When supplied, iorec's own
            version is stamped on a separate `writer_library` line instead, so both
            facts survive: what produced the mesh, and what serialised it. Has no
            effect if `metadata` is `None`, since no appendix is written in that case.
    """

    def write_geometry() -> None:
        if binary_mode:
            _save_2d_rec_binary(filename, points, segments, labels)
        else:
            _save_2d_rec_text(filename, points, segments, labels)

    _write_geometry_with_appendix(filename, write_geometry, metadata, written_by)


def _save_2d_rec_binary(
    filename: str | Path,
    points: NDArray[np.float64],
    segments: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> None:
    """Save a 2D rec file at filename. Binary only."""
    bytes_content = struct.pack("Q", len(points))
    bytes_content += points.flatten().tobytes()
    bytes_content += struct.pack("Q", len(segments))
    dt = np.dtype([("segments", np.int64, (2,)), ("labels", np.int32, (2,))])

    def map_tri_labels_func(
        i: int,
    ) -> tuple[Any, Any]:
        # -> tuple[np.int64, np.int64, np.int64, np.int64, np.int64]:
        return segments[i], labels[i]

    segments_and_labels = np.fromiter(
        (map_tri_labels_func(xi) for xi in range(len(segments))),
        dtype=dt,
        count=len(segments),
    )

    bytes_content += segments_and_labels.tobytes()
    with Path(filename).open("wb") as file:
        file.write(bytes_content)


def _save_2d_rec_text(
    filename: str | Path,
    points: NDArray[np.float64],
    segments: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> None:
    """Save a 2D rec file at filename. Text only."""
    content = ""
    content += str(len(points)) + "\n"

    for i in range(len(points)):
        content += f"{points[i][0]} {points[i][1]}\n"

    content += str(len(segments)) + "\n"

    for i in range(len(segments)):
        content += f"{segments[i][0]} {segments[i][1]} "
        content += f"{labels[i][0]} {labels[i][1]}\n"

    with Path(filename).open("w") as file:
        file.write(content)
