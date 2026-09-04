"""IO for .rec meshes."""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from iorec.rec_appendix import _iorec_version, _split_rec_bytes, write_rec_appendix


def _resolve_rec_path(filename: str | Path) -> Path:
    filename = Path(filename).resolve()
    if not filename.exists():
        error = f"{filename} not found."
        raise FileNotFoundError(error)
    if filename.suffix not in (".rec", ".arec"):
        error = f"{filename} is not a .rec or .arec file."
        raise ValueError(error)
    return filename


def load_rec(
    filename: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a rec mesh file.

    Args:
        filename (str | Path): The path to the file on disk.

    Raises:
        ValueError: If the path given is not a .rec or .arec file.
        FileNotFoundError: If the file doesn't exist.

    Returns:
        tuple[np.array, np.array, np.array]: (points, triangles, labels), with:
        - points: np.array of shape (num_points, 3) (float64)
        - triangles: np.array of shape (num_triangles, 3) (ulonglong)
        - labels: np.array of shape (num_triangles, 2) (uint)
    """
    filename = _resolve_rec_path(filename)
    return _load_rec_file(filename)


def load_rec_with_metadata(
    filename: str | Path,
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

    Raises:
        ValueError: If the path given is not a .rec or .arec file.
        FileNotFoundError: If the file doesn't exist.

    Returns:
        tuple[np.array, np.array, np.array, dict | None]: (points, triangles, labels,
        metadata), with points/triangles/labels as in `load_rec`, and metadata as in
        `load_rec_appendix` (`None` if the file has no appendix).
    """
    filename = _resolve_rec_path(filename)
    geometry_bytes, metadata = _split_rec_bytes(filename.read_bytes())

    if _is_binary_bytes(geometry_bytes):
        points, triangles, labels = _parse_binary_rec_bytes(geometry_bytes)
    else:
        points, triangles, labels = _parse_text_rec_bytes(geometry_bytes)

    return points, triangles, labels, metadata


def _load_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    if _is_binary(filename):
        return _load_binary_rec_file(filename)
    else:
        return _load_text_rec_file(filename)


def _load_binary_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    with Path.open(filename, "rb") as f:
        # First we read the number of vertices
        (number_of_points,) = struct.unpack("Q", f.read(8))
        # Then we read the vertices : they are in a 3D space
        points = np.fromfile(f, count=3 * number_of_points, dtype=np.float64).reshape(
            (number_of_points, 3),
        )

        # Number of triangles
        (number_of_triangles,) = struct.unpack("Q", f.read(8))
        # Each line defines a triangle and 2 regions labels
        dtype_triangles_and_labels = np.dtype(
            [("triangles", np.int64, (3,)), ("labels", np.int32, (2,))],
        )
        triangles_and_labels = np.fromfile(
            f,
            count=number_of_triangles,
            dtype=dtype_triangles_and_labels,
        )

    return (
        points,
        triangles_and_labels["triangles"].astype(np.int64),
        triangles_and_labels["labels"].astype(np.int64),
    )


def _load_text_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    with Path.open(filename) as f:
        # First we read the number of vertices
        number_of_points = int(f.readline().strip())
        # Then we read the vertices : they are in a 3D space
        points = (
            np.loadtxt(f, max_rows=number_of_points, dtype=np.float64, ndmin=2)
            if number_of_points > 0
            else np.empty((0, 3), dtype=np.float64)
        )

        # Number of triangles
        number_of_triangles = int(f.readline().strip())
        # Each line defines a triangle and 2 regions labels
        triangles_and_labels = (
            np.loadtxt(f, max_rows=number_of_triangles, dtype=np.int64, ndmin=2)
            if number_of_triangles > 0
            else np.empty((0, 5), dtype=np.int64)
        )

    return (
        points,
        triangles_and_labels[:, 0:3],
        triangles_and_labels[:, 3:5],
    )


def _is_binary_bytes(data: bytes, chunk_size: int = 64) -> bool:
    return b"\x00" in data[:chunk_size]


def _is_binary(file_name: Path, chunk_size: int = 64) -> bool:
    with Path.open(file_name, "rb") as f:
        return _is_binary_bytes(f.read(chunk_size), chunk_size)


def _parse_binary_rec_bytes(
    data: bytes,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Parse geometry from an in-memory buffer, e.g. the geometry half of a buffer
    already split by `iorec.rec_appendix._split_rec_bytes`.

    Kept separate from `_load_binary_rec_file` (which streams from disk with
    `np.fromfile`) rather than unifying the two: unifying would make plain `load_rec`
    hold the whole file in memory as `bytes` *and* as the parsed array at once, which
    for very large files is a real peak-memory regression that `load_rec` shouldn't
    pay just so `load_rec_with_metadata` can avoid a second read.
    """
    (number_of_points,) = struct.unpack_from("Q", data)
    offset = 8
    points_count = 3 * number_of_points
    points = (
        np.frombuffer(data, dtype=np.float64, count=points_count, offset=offset)
        .reshape((number_of_points, 3))
        .copy()
    )
    offset += points_count * 8

    (number_of_triangles,) = struct.unpack_from("Q", data, offset)
    offset += 8
    dtype_triangles_and_labels = np.dtype(
        [("triangles", np.int64, (3,)), ("labels", np.int32, (2,))],
    )
    triangles_and_labels = np.frombuffer(
        data,
        dtype=dtype_triangles_and_labels,
        count=number_of_triangles,
        offset=offset,
    )

    return (
        points,
        triangles_and_labels["triangles"].astype(np.int64),
        triangles_and_labels["labels"].astype(np.int64),
    )


def _parse_text_rec_bytes(
    data: bytes,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Parse geometry from an in-memory buffer. See `_parse_binary_rec_bytes` for why
    this is kept separate from `_load_text_rec_file` rather than unifying the two.
    """
    with io.StringIO(data.decode()) as f:
        number_of_points = int(f.readline().strip())
        points = (
            np.loadtxt(f, max_rows=number_of_points, dtype=np.float64, ndmin=2)
            if number_of_points > 0
            else np.empty((0, 3), dtype=np.float64)
        )

        number_of_triangles = int(f.readline().strip())
        triangles_and_labels = (
            np.loadtxt(f, max_rows=number_of_triangles, dtype=np.int64, ndmin=2)
            if number_of_triangles > 0
            else np.empty((0, 5), dtype=np.int64)
        )

    return (
        points,
        triangles_and_labels[:, 0:3],
        triangles_and_labels[:, 3:5],
    )


def save_rec(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    binary_mode: bool = False,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a rec mesh file at filename. Binary or text mode are available.

    Binary mode is more compact but might not be universally understood.

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
    """
    appendix_bytes = (
        write_rec_appendix(metadata, written_by=_iorec_version())
        if metadata is not None
        else None
    )

    if binary_mode:
        _save_rec_binary(filename, points, triangles, labels)
    else:
        _save_rec_text(filename, points, triangles, labels)

    if appendix_bytes is not None:
        with Path(filename).open("ab") as file:
            file.write(appendix_bytes)


def _save_rec_binary(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> None:
    """Save a rec file at filename. Binary only."""
    bytes_content = struct.pack("Q", len(points))
    bytes_content += points.flatten().tobytes()
    bytes_content += struct.pack("Q", len(triangles))
    dt = np.dtype([("triangles", np.int64, (3,)), ("labels", np.int32, (2,))])

    def map_tri_labels_func(
        i: int,
    ) -> tuple[Any, Any]:
        # -> tuple[np.int64, np.int64, np.int64, np.int64, np.int64]:
        return triangles[i], labels[i]

    triangles_and_labels = np.fromiter(
        (map_tri_labels_func(xi) for xi in np.arange(len(triangles))),
        dtype=dt,
        count=len(triangles),
    )

    bytes_content += triangles_and_labels.tobytes()
    with Path(filename).open("wb") as file:
        file.write(bytes_content)


def _save_rec_text(
    filename: str | Path,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> None:
    """Save a rec file at filename. Text only."""
    content = ""
    content += str(len(points)) + "\n"

    for i in range(len(points)):
        # content += f"{points[i][0]:.5f} {points[i][1]:.5f} {points[i][2]:.5f}\n"
        content += f"{points[i][0]} {points[i][1]} {points[i][2]}\n"

    content += str(len(triangles)) + "\n"

    for i in range(len(triangles)):
        content += f"{triangles[i][0]} {triangles[i][1]} {triangles[i][2]} "
        content += f"{labels[i][0]} {labels[i][1]}\n"

    with Path(filename).open("w") as file:
        file.write(content)
