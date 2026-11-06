# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2023-2026 the iorec authors
"""Opt-in structural checks for .rec/.arec files, run by the readers on `validate=True`.

Without `validate=True` the readers keep their historical, permissive behaviour: a
truncated file, a 2D file read as 3D, or out-of-range indices are returned as they come.
With it, any such problem raises `RecFormatError`, a `ValueError` subclass, naming the
file and the first offending record.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from iorec.rec_appendix import _appendix_marker


class MeshFormatError(ValueError):
    """Raised by the readers for a malformed or unsupported mesh file, of any format."""


class RecFormatError(MeshFormatError):
    """Raised by the readers, with `validate=True`, for a malformed .rec/.arec file."""


def _binary_geometry_size(
    number_of_points: int,
    number_of_cells: int,
    dimension: int,
    record_itemsize: int,
) -> int:
    """Return the exact byte size of a binary geometry block (without appendix)."""
    return 8 + 8 * dimension * number_of_points + 8 + record_itemsize * number_of_cells


def _check_binary_room(
    filename: Path | str,
    available: int,
    required: int,
    what: str,
) -> None:
    """Raise if fewer than `required` bytes are available for the block `what`."""
    if required > available:
        error = (
            f"{filename}: truncated binary {what} "
            f"({available} bytes available, {required} needed)."
        )
        raise RecFormatError(error)


def _check_binary_tail(filename: Path | str, tail: bytes) -> None:
    """Raise unless the bytes after the geometry are empty or a metadata appendix."""
    if tail and not tail.startswith(_appendix_marker()):
        error = (
            f"{filename}: {len(tail)} unexpected trailing bytes after the binary "
            "geometry (not a rec-appendix block)."
        )
        raise RecFormatError(error)


def _check_text_tail(filename: Path | str, tail: str) -> None:
    """Raise unless the text after the geometry is blank or a metadata appendix."""
    stripped = tail.lstrip()
    if stripped and not stripped.startswith(_appendix_marker().decode("ascii")):
        error = (
            f"{filename}: unexpected content after the last record "
            "(not a rec-appendix block)."
        )
        raise RecFormatError(error)


def _parse_count(filename: Path | str, line: str, what: str) -> int:
    """Parse a text count line strictly."""
    try:
        value = int(line.strip())
    except ValueError as exc:
        error = f"{filename}: invalid {what} count {line.strip()!r}."
        raise RecFormatError(error) from exc
    if value < 0:
        error = f"{filename}: {what} count must be non-negative, got {value}."
        raise RecFormatError(error)
    return value


def _check_mesh(
    filename: Path | str,
    points: NDArray[np.float64],
    cells: NDArray[np.int64],
    labels: NDArray[np.int64],
    dimension: int,
    counts: tuple[int, int],
    *,
    error_type: type[ValueError] = RecFormatError,
) -> None:
    """Check shapes, coordinates, indices and labels of a parsed mesh.

    Args:
        filename: used in error messages only.
        points: parsed points, expected shape `(counts[0], dimension)`.
        cells: parsed triangles (3D) or segments (2D), expected shape
            `(counts[1], dimension)`.
        labels: parsed labels, expected shape `(counts[1], 2)`.
        dimension: 3 for triangle meshes, 2 for segment meshes.
        counts: the expected point and cell counts, e.g. those declared in the file.
        error_type: the exception raised, `RecFormatError` for .rec/.arec files.

    Raises:
        ValueError: `error_type`, on the first violated condition.
    """
    number_of_points, number_of_cells = counts
    cell_name = "triangle" if dimension == 3 else "segment"
    expected = {
        "points": (points, (number_of_points, dimension)),
        f"{cell_name}s": (cells, (number_of_cells, dimension)),
        "labels": (labels, (number_of_cells, 2)),
    }
    for name, (array, shape) in expected.items():
        if array.shape != shape:
            error = (
                f"{filename}: {name} have shape {array.shape}, expected {shape} "
                f"(expected {number_of_points} points and {number_of_cells} "
                f"{cell_name}s in {dimension}D)."
            )
            raise error_type(error)

    not_finite = np.flatnonzero(~np.isfinite(points).all(axis=1))
    if not_finite.size:
        error = f"{filename}: point {not_finite[0]} has a non-finite coordinate."
        raise error_type(error)

    if number_of_cells:
        out_of_range = np.flatnonzero(
            ((cells < 0) | (cells >= number_of_points)).any(axis=1),
        )
        if out_of_range.size:
            first = out_of_range[0]
            error = (
                f"{filename}: {cell_name} {first} has a vertex index outside "
                f"[0, {number_of_points}): {cells[first].tolist()}."
            )
            raise error_type(error)

        negative = np.flatnonzero((labels < 0).any(axis=1))
        if negative.size:
            first = negative[0]
            error = (
                f"{filename}: {cell_name} {first} has a negative label: "
                f"{labels[first].tolist()}."
            )
            raise error_type(error)

        same = np.flatnonzero(labels[:, 0] == labels[:, 1])
        if same.size:
            first = same[0]
            error = (
                f"{filename}: {cell_name} {first} separates label "
                f"{labels[first, 0]} from itself."
            )
            raise error_type(error)


def _check_triangle_mesh(
    filename: Path | str,
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> None:
    """Check a parsed triangle mesh of a format that declares no counts.

    Raises:
        MeshFormatError: if shapes, coordinates, indices or labels are invalid.
    """
    _check_mesh(
        filename,
        points,
        triangles,
        labels,
        3,
        (len(points), len(triangles)),
        error_type=MeshFormatError,
    )
