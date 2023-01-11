"""
Open MultiTracker files utility.
Define one function: open_rec
- Input
    - filename: str
- Output
    - points: np.array of shape (num_points, 3) (float64)
    - triangles: np.array of shape (num_triangles, 3) (uint64)
    - labels: np.array of shape (num_triangles, 2) (uint64)
- Raises
    - ValueError if the filename is not a .rec or . arec file.
"""
import struct
from pathlib import Path
import numpy as np


def open_rec(filename: str | Path) -> tuple[np.array, np.array, np.array] | ValueError | FileNotFoundError:
    """Read a MultiTracker file.

    Args:
        filename (str | Path): The path to the file on disk.

    Raises:
        ValueError: If the path given is not a .rec or .arec file.
        FileNotFoundError: If the file doesn't exist.

    Returns:
        tuple[np.array, np.array, np.array]: (points, triangles, labels), with:
        - points: np.array of shape (num_points, 3) (float64)
        - triangles: np.array of shape (num_triangles, 3) (uint64)
        - labels: np.array of shape (num_triangles, 2) (uint64)
    """
    filename = Path(filename).resolve()
    if not filename.exists():
        raise FileNotFoundError(f"{filename} not found.")
    if filename.suffix == ".rec" or filename.suffix == ".arec":
        return _open_rec_file(filename)
    else:
        raise ValueError(f"{filename} is not a .rec or .arec file.")


def _open_rec_file(filename: Path):
    if _is_binary(filename):
        return _open_binary_rec_file(filename)
    else:
        return _open_text_rec_file(filename)


def _open_binary_rec_file(filename: Path):
    with open(filename, 'rb') as f:
        # First we read the number of vertices
        number_of_points, = struct.unpack('Q', f.read(8))
        # Then we read the vertices : they are in a 3D space
        points = np.fromfile(f, count=3 * number_of_points,
                             dtype=np.float64).reshape((number_of_points, 3))

        # Number of triangles
        number_of_triangles, = struct.unpack('Q', f.read(8))
        # Each line defines a triangle and 2 regions labels
        dtype_triangles_and_labels = np.dtype(
            [('triangles', np.uint64, (3,)), ('labels', np.uint64, (2,))])
        triangles_and_labels = np.fromfile(
            f, count=number_of_triangles, dtype=dtype_triangles_and_labels)

    return (points, triangles_and_labels['triangles'], triangles_and_labels['labels'])


def _open_text_rec_file(filename: Path):
    with open(filename, 'r') as f:
        # First we read the number of vertices
        number_of_points = int(f.readline().strip())
        # Then we read the vertices : they are in a 3D space
        points = np.fromfile(
            f, count=3 * number_of_points, sep="\n", dtype=np.float64).reshape((number_of_points, 3))

        # Number of triangles
        number_of_triangles = int(f.readline().strip())
        # Each line defines a triangle and 2 regions labels
        triangles_and_labels = np.fromfile(
            f, count=5 * number_of_triangles, sep="\n", dtype=np.uint64).reshape((number_of_triangles, 5))

    return (points, triangles_and_labels[:, 0:3], triangles_and_labels[:, 3:5])


def _is_binary(file_name: Path):
    try:
        with open(file_name, 'tr') as check_file:  # try open file in text mode
            check_file.read()
            return False
    except IOError as fee:
        print(fee)
        raise
    except:  # noqa # if fail then file is non-text (binary).. Not perfect but should work in most cases
        return True
