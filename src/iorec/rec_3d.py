# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2023-2026 the iorec authors
"""IO for .rec meshes."""

from __future__ import annotations

import io
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from numpy.typing import NDArray

from iorec._utils import (
    _fast_text_rec,
    _fast_text_rec_file,
    _format_text_rec,
    _is_binary,
    _is_binary_bytes,
    _read_text_count,
    _read_text_rows,
    _resolve_rec_path,
    _write_geometry_with_appendix,
)
from iorec._validation import (
    RecFormatError,
    _binary_geometry_size,
    _check_binary_room,
    _check_binary_tail,
    _check_mesh,
    _check_text_tail,
)
from iorec.rec_appendix import (
    _DEFAULT_APPENDIX_VERSION,
    _appendix_marker,
    _split_rec_bytes,
)


def load_rec(
    filename: str | Path,
    *,
    validate: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a rec mesh file.

    Args:
        filename (str | Path): The path to the file on disk.
        validate (bool, optional): Check the file's structure and content: binary
            size consistent with the declared counts (a trailing metadata appendix
            is allowed), the expected number of rows and columns in text files,
            finite coordinates, vertex indices in range, non-negative labels and two
            different labels on each triangle. Defaults to False, which reads the
            file as earlier versions did, without these checks.

    Raises:
        ValueError: If the path given is not a .rec or .arec file.
        FileNotFoundError: If the file doesn't exist.
        RecFormatError: With `validate=True`, if the file is malformed (a
            `ValueError` subclass).

    Returns:
        tuple[np.array, np.array, np.array]: (points, triangles, labels), with:
        - points: np.array of shape (num_points, 3) (float64)
        - triangles: np.array of shape (num_triangles, 3) (int64)
        - labels: np.array of shape (num_triangles, 2) (int64)
    """
    filename = _resolve_rec_path(filename)
    return _load_rec_file(filename, validate=validate)


def load_rec_with_metadata(
    filename: str | Path,
    *,
    validate: bool = False,
    metadata_types: Mapping[str, object] | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    dict[str, Any] | None,
]:
    """Read a rec mesh file together with its metadata appendix, if any.

    Reads the file once and splits geometry from metadata in memory -- unlike calling
    `load_rec` and `load_rec_appendix` separately, which would each read the whole file
    on their own.

    Args:
        filename (str | Path): The path to the file on disk.
        validate (bool, optional): Check the geometry as `load_rec` does with
            `validate=True`. Defaults to False.
        metadata_types: version-1 appendices only, the type of named keys, as in
            `load_rec_appendix`.

    Raises:
        ValueError: If the path given is not a .rec or .arec file.
        FileNotFoundError: If the file doesn't exist.
        RecFormatError: With `validate=True`, if the file is malformed (a
            `ValueError` subclass).

    Returns:
        tuple[np.array, np.array, np.array, dict | None]: (points, triangles, labels,
        metadata), with points/triangles/labels as in `load_rec`, and metadata as in
        `load_rec_appendix` (`None` if the file has no appendix).
    """
    filename = _resolve_rec_path(filename)
    geometry_bytes, metadata = _split_rec_bytes(
        filename.read_bytes(),
        metadata_types,
    )

    if _is_binary_bytes(geometry_bytes):
        points, triangles, labels = _parse_binary_rec_bytes(
            geometry_bytes,
            validate=validate,
            filename=filename,
        )
    else:
        points, triangles, labels = _parse_text_rec_bytes(
            geometry_bytes,
            validate=validate,
            filename=filename,
        )

    return points, triangles, labels, metadata


# On-disk layout of binary files: little-endian whatever the machine. Existing files
# come from little-endian machines (x86-64, arm64), where this changes nothing; on a
# big-endian machine, files stay readable and portable instead of byte-swapped.
_DTYPE_TRIANGLES_AND_LABELS = np.dtype(
    [("triangles", "<i8", (3,)), ("labels", "<i4", (2,))],
)
_DTYPE_POINTS = np.dtype("<f8")


def _load_rec_file(
    filename: Path,
    *,
    validate: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    if _is_binary(filename):
        return _load_binary_rec_file(filename, validate=validate)
    else:
        return _load_text_rec_file(filename, validate=validate)


def _load_binary_rec_file(
    filename: Path,
    *,
    validate: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    size = filename.stat().st_size if validate else 0
    with Path.open(filename, "rb") as f:
        # First we read the number of vertices
        header = f.read(8)
        if validate:
            _check_binary_room(filename, len(header), 8, "point count")
        (number_of_points,) = struct.unpack("<Q", header)
        if validate:
            _check_binary_room(filename, size - 8, 24 * number_of_points + 8, "points")
        # Then we read the vertices : they are in a 3D space
        points = (
            np.fromfile(f, count=3 * number_of_points, dtype=_DTYPE_POINTS)
            .reshape((number_of_points, 3))
            .astype(np.float64, copy=False)
        )

        # Number of triangles
        (number_of_triangles,) = struct.unpack("<Q", f.read(8))
        geometry_size = _binary_geometry_size(
            number_of_points,
            number_of_triangles,
            3,
            _DTYPE_TRIANGLES_AND_LABELS.itemsize,
        )
        if validate:
            _check_binary_room(filename, size, geometry_size, "triangles")
        # Each line defines a triangle and 2 regions labels
        triangles_and_labels = np.fromfile(
            f,
            count=number_of_triangles,
            dtype=_DTYPE_TRIANGLES_AND_LABELS,
        )
        if validate:
            _check_binary_tail(filename, f.read(len(_appendix_marker())))

    triangles = triangles_and_labels["triangles"].astype(np.int64)
    labels = triangles_and_labels["labels"].astype(np.int64)
    if validate:
        counts = (number_of_points, number_of_triangles)
        _check_mesh(filename, points, triangles, labels, 3, counts)
    return points, triangles, labels


def _load_text_rec_file(
    filename: Path,
    *,
    validate: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    fast = _fast_text_rec_file(filename, 3, validate=validate)
    if fast is not None:
        return fast
    with Path.open(filename) as f:
        return _read_text_rec(f, filename, validate=validate)


def _read_text_rec(
    f: TextIO,
    filename: Path | str,
    *,
    validate: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read text-mode geometry from an open stream (a file or an in-memory buffer)."""
    # First we read the number of vertices
    number_of_points = _read_text_count(f, filename, "point", validate=validate)
    # Then we read the vertices : they are in a 3D space
    points = _read_text_rows(
        f,
        number_of_points,
        np.float64,
        3,
        filename,
        "point",
        validate=validate,
    )

    # Number of triangles
    number_of_triangles = _read_text_count(f, filename, "triangle", validate=validate)
    # Each line defines a triangle and 2 regions labels
    triangles_and_labels = _read_text_rows(
        f,
        number_of_triangles,
        np.int64,
        5,
        filename,
        "triangle",
        validate=validate,
    )
    triangles = triangles_and_labels[:, 0:3]
    labels = triangles_and_labels[:, 3:5]

    if validate:
        if triangles_and_labels.shape[1:] != (5,):
            error = (
                f"{filename}: triangle records have {triangles_and_labels.shape[1]} "
                "columns, expected 5 (three vertex indices and two labels)."
            )
            raise RecFormatError(error)
        _check_text_tail(filename, f.read())
        counts = (number_of_points, number_of_triangles)
        _check_mesh(filename, points, triangles, labels, 3, counts)
    return points, triangles, labels


def _parse_binary_rec_bytes(
    data: bytes,
    *,
    validate: bool = False,
    filename: Path | str = "<buffer>",
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Parse geometry from an in-memory buffer.

    E.g. the geometry half of a buffer already split by
    `iorec.rec_appendix._split_rec_bytes`.

    Kept separate from `_load_binary_rec_file` (which streams from disk with
    `np.fromfile`) rather than unifying the two: unifying would make plain `load_rec`
    hold the whole file in memory as `bytes` *and* as the parsed array at once, which
    for very large files is a real peak-memory regression that `load_rec` shouldn't
    pay just so `load_rec_with_metadata` can avoid a second read.
    """
    if validate:
        _check_binary_room(filename, len(data), 8, "point count")
    (number_of_points,) = struct.unpack_from("<Q", data)
    offset = 8
    points_count = 3 * number_of_points
    if validate:
        _check_binary_room(filename, len(data) - 8, 8 * points_count + 8, "points")
    points = (
        np.frombuffer(data, dtype=_DTYPE_POINTS, count=points_count, offset=offset)
        .reshape((number_of_points, 3))
        .astype(np.float64)  # a native-endian copy, not a view on `data`
    )
    offset += points_count * 8

    (number_of_triangles,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    if validate:
        geometry_size = _binary_geometry_size(
            number_of_points,
            number_of_triangles,
            3,
            _DTYPE_TRIANGLES_AND_LABELS.itemsize,
        )
        _check_binary_room(filename, len(data), geometry_size, "triangles")
        _check_binary_tail(filename, data[geometry_size:])
    triangles_and_labels = np.frombuffer(
        data,
        dtype=_DTYPE_TRIANGLES_AND_LABELS,
        count=number_of_triangles,
        offset=offset,
    )

    triangles = triangles_and_labels["triangles"].astype(np.int64)
    labels = triangles_and_labels["labels"].astype(np.int64)
    if validate:
        counts = (number_of_points, number_of_triangles)
        _check_mesh(filename, points, triangles, labels, 3, counts)
    return points, triangles, labels


def _parse_text_rec_bytes(
    data: bytes,
    *,
    validate: bool = False,
    filename: Path | str = "<buffer>",
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Parse geometry from an in-memory buffer.

    Shares `_read_text_rec` with `_load_text_rec_file`: text parsing reads line by line
    from a stream in both cases, so there is no peak-memory reason to keep two copies.
    """
    fast = _fast_text_rec(data, filename, 3, validate=validate)
    if fast is not None:
        return fast
    with io.StringIO(data.decode()) as f:
        return _read_text_rec(f, filename, validate=validate)


def save_rec(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    binary_mode: bool = False,
    *,
    metadata: dict[str, Any] | None = None,
    written_by: str | None = None,
    appendix_version: int = _DEFAULT_APPENDIX_VERSION,
) -> None:
    """Save a rec mesh file at filename. Binary or text mode are available.

    Binary mode is more compact but might not be universally understood.

    Both modes store every value exactly. Binary mode writes float64 coordinates,
    int64 indices and int32 labels, little-endian on every machine, converting arrays
    of other dtypes. Text mode writes each value with `repr`, the shortest text that
    reads back to the same float64.

    Args:
        filename (str | Path): where to save the mesh file.
        points (NDArray[np.float64]): Mesh points.
        triangles (NDArray[np.int64]): Mesh triangles (topology).
        labels (NDArray[np.int64]): Triangles multimaterial labels.
        binary_mode (bool, optional): Choose binary mode. Defaults to False.
        metadata (dict[str, Any] | None, optional): appendix fields (e.g.
            `coordinate_frame`, `axis_order`, `spacing`, `spacing_unit`,
            `source_shape`, `crop_offset`). Defaults to `None`, which writes nothing
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
        appendix_version (int, optional): appendix format, 1 (`key = text` lines)
            or 2 (`key: JSON` lines, every value typed). Defaults to 2. No effect if
            `metadata` is `None`.
    """

    def write_geometry() -> None:
        if binary_mode:
            _save_rec_binary(filename, points, triangles, labels)
        else:
            _save_rec_text(filename, points, triangles, labels)

    _write_geometry_with_appendix(
        filename,
        write_geometry,
        metadata,
        written_by,
        appendix_version,
    )


def _save_rec_binary(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> None:
    """Save a rec file at filename. Binary only."""
    records = np.empty(len(triangles), dtype=_DTYPE_TRIANGLES_AND_LABELS)
    records["triangles"] = triangles
    records["labels"] = labels

    bytes_content = struct.pack("<Q", len(points))
    bytes_content += np.ascontiguousarray(points, dtype=_DTYPE_POINTS).tobytes()
    bytes_content += struct.pack("<Q", len(triangles))
    bytes_content += records.tobytes()
    with Path(filename).open("wb") as file:
        file.write(bytes_content)


def _save_rec_text(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> None:
    """Save a rec file at filename. Text only."""
    content = _format_text_rec(points, triangles, labels, 3)

    with Path(filename).open("w") as file:
        file.write(content)
