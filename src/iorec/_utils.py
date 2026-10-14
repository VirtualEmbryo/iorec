"""Shared helpers for .rec/.arec files, used by both the 2D and 3D readers."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, TextIO

import numpy as np
from numpy.typing import NDArray

from iorec._fast_text import parse_text_geometry
from iorec._validation import (
    RecFormatError,
    _check_mesh,
    _check_text_tail,
    _parse_count,
)
from iorec.rec_appendix import _iorec_version, write_rec_appendix


def _resolve_rec_path(filename: str | Path) -> Path:
    filename = Path(filename).resolve()
    if not filename.exists():
        error = f"{filename} not found."
        raise FileNotFoundError(error)
    if filename.suffix not in (".rec", ".arec"):
        error = f"{filename} is not a .rec or .arec file."
        raise ValueError(error)
    return filename


def _is_binary_bytes(data: bytes, chunk_size: int = 64) -> bool:
    return b"\x00" in data[:chunk_size]


def _is_binary(file_name: Path, chunk_size: int = 64) -> bool:
    with Path.open(file_name, "rb") as f:
        return _is_binary_bytes(f.read(chunk_size), chunk_size)


def _write_geometry_with_appendix(
    filename: str | Path,
    write_geometry: Callable[[], None],
    metadata: dict[str, Any] | None,
    written_by: str | None = None,
) -> None:
    """Write geometry, then an optional trailing metadata appendix.

    `write_geometry` must write the full geometry file at `filename` (binary or text
    mode); the appendix, if any, is appended afterwards so it stays trailing.

    `written_by` names the program that produced the mesh. When `None`, iorec's own
    version is stamped instead, unchanged from before this parameter existed. When
    given, iorec's own version is stamped on a separate `writer_library` line instead,
    so both facts survive.
    """
    appendix_bytes = None
    if metadata is not None:
        if written_by is None:
            appendix_bytes = write_rec_appendix(metadata, written_by=_iorec_version())
        else:
            appendix_bytes = write_rec_appendix(
                metadata,
                written_by=written_by,
                writer_library=f"iorec {_iorec_version()}",
            )

    write_geometry()

    if appendix_bytes is not None:
        with Path(filename).open("ab") as file:
            file.write(appendix_bytes)


# Text files smaller than this are read by the general reader: the first import of
# pyarrow costs more than parsing them.
_FAST_TEXT_MIN_BYTES = 256 * 1024


def _fast_text_rec(
    data: bytes,
    filename: Path | str,
    dimension: int,
    *,
    validate: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]] | None:
    """Parse text geometry with pyarrow, or return `None` for the general reader.

    `None` means pyarrow is missing, the buffer is smaller than the size threshold, or
    it is not plain well-formed text (any case the general reader accepts with another
    result, or rejects, is left to it).

    Args:
        data: file content, geometry then optional appendix.
        filename: used in error messages only.
        dimension: 3 for triangle meshes, 2 for segment meshes.
        validate: run the checks of `validate=True`.

    Raises:
        RecFormatError: with `validate`, if the parsed mesh is malformed.
    """
    if len(data) < _FAST_TEXT_MIN_BYTES:
        return None
    parsed = parse_text_geometry(data, dimension, dimension + 2)
    if parsed is None:
        return None
    points, records, end = parsed
    cells = records[:, 0:dimension]
    labels = records[:, dimension : dimension + 2]
    try:
        tail = data[end:].decode()
    except UnicodeDecodeError:
        return None
    if validate:
        _check_text_tail(filename, tail)
        _check_mesh(
            filename,
            points,
            cells,
            labels,
            dimension,
            (len(points), len(records)),
        )
    return points, cells, labels


def _fast_text_rec_file(
    filename: Path,
    dimension: int,
    *,
    validate: bool,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]] | None:
    """Read a file for `_fast_text_rec`, skipping files below the size threshold."""
    if filename.stat().st_size < _FAST_TEXT_MIN_BYTES:
        return None
    return _fast_text_rec(filename.read_bytes(), filename, dimension, validate=validate)


def _read_text_count(
    f: TextIO,
    filename: Path | str,
    what: str,
    *,
    validate: bool,
) -> int:
    """Read one count line of a text .rec file."""
    line = f.readline()
    if validate:
        return _parse_count(filename, line, what)
    return int(line.strip())


def _read_text_rows(
    f: TextIO,
    count: int,
    dtype: type[np.float64] | type[np.int64],
    columns: int,
    filename: Path | str,
    what: str,
    *,
    validate: bool,
) -> NDArray[Any]:
    """Read `count` rows of a text .rec block, stopping exactly after the last one.

    Without `validate`, behaviour is unchanged from earlier versions: `np.loadtxt`
    errors propagate as they are, and a short block is returned short. With
    `validate`, parse errors become `RecFormatError` and the caller checks the shape.
    """
    if count <= 0:
        return np.empty((0, columns), dtype=dtype)
    if not validate:
        return np.loadtxt(f, max_rows=count, dtype=dtype, ndmin=2)
    try:
        with warnings.catch_warnings():
            # An empty block is reported by the shape check, with the file name.
            warnings.simplefilter("ignore", UserWarning)
            return np.loadtxt(f, max_rows=count, dtype=dtype, ndmin=2)
    except ValueError as exc:
        error = f"{filename}: invalid {what} block: {exc}"
        raise RecFormatError(error) from exc


def _format_text_rec(
    points: NDArray[Any],
    cells: NDArray[Any],
    labels: NDArray[Any],
    dimension: int,
) -> str:
    """Return the text of a .rec file: counts, one row per point and per cell.

    Each value is written with `repr` (the shortest text that reads back to the same
    float64, or the integer's digits), exactly as formatting each NumPy scalar in an
    f-string did in earlier versions; `tolist()` turns the scalars into Python numbers
    without changing their values. Each block is built with one `%` operation instead
    of a Python loop over rows.
    """
    points = np.asarray(points)[:, :dimension]
    records = np.empty((len(cells), dimension + 2), dtype=object)
    records[:, :dimension] = np.asarray(cells)[:, :dimension]
    records[:, dimension:] = np.asarray(labels)[:, :2]
    point_row = " ".join(["%r"] * dimension) + "\n"
    record_row = " ".join(["%r"] * (dimension + 2)) + "\n"
    return (
        f"{len(points)}\n"
        + (point_row * len(points)) % tuple(points.reshape(-1).tolist())
        + f"{len(records)}\n"
        + (record_row * len(records)) % tuple(records.reshape(-1).tolist())
    )
