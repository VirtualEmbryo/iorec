"""IO for 2D .rec meshes."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from iorec.rec_appendix import _iorec_version, write_rec_appendix


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
    filename = Path(filename).resolve()
    if not filename.exists():
        error = f"{filename} not found."
        raise FileNotFoundError(error)
    if filename.suffix == ".rec" or filename.suffix == ".arec":
        return _load_2d_rec_file(filename)
    else:
        error = f"{filename} is not a .rec or .arec file."
        raise ValueError(error)


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


def _is_binary(file_name: Path, chunk_size: int = 64) -> bool:
    with Path.open(file_name, "rb") as f:
        return b"\x00" in f.read(chunk_size)


def save_2d_rec(
    filename: str | Path,
    points: NDArray[np.float64],
    segments: NDArray[np.int64],
    labels: NDArray[np.int64],
    binary_mode: bool = False,
    *,
    metadata: dict[str, Any] | None = None,
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
    """
    appendix_bytes = (
        write_rec_appendix(metadata, written_by=_iorec_version())
        if metadata is not None
        else None
    )

    if binary_mode:
        _save_2d_rec_binary(filename, points, segments, labels)
    else:
        _save_2d_rec_text(filename, points, segments, labels)

    if appendix_bytes is not None:
        with Path(filename).open("ab") as file:
            file.write(appendix_bytes)


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
        (map_tri_labels_func(xi) for xi in np.arange(len(segments))),
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
