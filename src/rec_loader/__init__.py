"""
Open MultiTracker files utility.
Define one function: open_rec
- Input
    - filename: str
- Output
    - points: np.array of shape (num_points, 3) (float64)
    - triangles: np.array of shape (num_triangles, 3) (ulonglong)
    - labels: np.array of shape (num_triangles, 2) (uint)
- Raises
    - ValueError if the filename is not a .rec or . arec file.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def open_rec(
    filename: str | Path,
) -> tuple[NDArray[np.float64], NDArray[np.ulonglong], NDArray[np.uint]]:
    """Read a MultiTracker file.

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
    filename = Path(filename).resolve()
    if not filename.exists():
        error = f"{filename} not found."
        raise FileNotFoundError(error)
    if filename.suffix == ".rec" or filename.suffix == ".arec":
        return _open_rec_file(filename)
    else:
        error = f"{filename} is not a .rec or .arec file."
        raise ValueError(error)


def _open_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.ulonglong], NDArray[np.uint]]:
    if _is_binary(filename):
        return _open_binary_rec_file(filename)
    else:
        return _open_text_rec_file(filename)


def _open_binary_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.ulonglong], NDArray[np.uint]]:
    with Path.open(filename, "rb") as f:
        # First we read the number of vertices
        (number_of_points,) = struct.unpack("Q", f.read(8))
        # Then we read the vertices : they are in a 3D space
        points = np.fromfile(f, count=3 * number_of_points, dtype=np.float64).reshape(
            (number_of_points, 3)
        )

        # Number of triangles
        (number_of_triangles,) = struct.unpack("Q", f.read(8))
        # Each line defines a triangle and 2 regions labels
        dtype_triangles_and_labels = np.dtype(
            [("triangles", np.int64, (3,)), ("labels", np.int32, (2,))]
        )
        triangles_and_labels = np.fromfile(
            f, count=number_of_triangles, dtype=dtype_triangles_and_labels
        )

    return (
        points,
        triangles_and_labels["triangles"].astype(np.ulonglong),
        triangles_and_labels["labels"].astype(np.uint),
    )


def _open_text_rec_file(
    filename: Path,
) -> tuple[NDArray[np.float64], NDArray[np.ulonglong], NDArray[np.uint]]:
    with Path.open(filename) as f:
        # First we read the number of vertices
        number_of_points = int(f.readline().strip())
        # Then we read the vertices : they are in a 3D space
        points = np.fromfile(
            f, count=3 * number_of_points, sep="\n", dtype=np.float64
        ).reshape((number_of_points, 3))

        # Number of triangles
        number_of_triangles = int(f.readline().strip())
        # Each line defines a triangle and 2 regions labels
        triangles_and_labels = np.fromfile(
            f, count=5 * number_of_triangles, sep="\n", dtype=np.uint64
        ).reshape(
            (number_of_triangles, 5),
        )

    return (
        points,
        triangles_and_labels[:, 0:3].astype(np.ulonglong),
        triangles_and_labels[:, 3:5].astype(np.uint),
    )


def _is_binary(file_name: Path) -> bool:
    try:
        with Path.open(file_name, encoding="utf-8") as f:
            f.read()
            return False
    except UnicodeDecodeError:
        return True
