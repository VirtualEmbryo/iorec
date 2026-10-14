"""Optional fast parser for the geometry of text .rec/.arec files, based on pyarrow."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Characters of a plain float block (numbers, `nan`, `inf`) and of an integer block.
# Any other byte (tab, carriage return, `#`, letters of hexadecimal floats...) sends
# the file to the general reader.
_FLOAT_BYTES = b"0123456789+-.eE naifNAIF\n"
_INT_BYTES = b"0123456789+- \n"


@functools.lru_cache(maxsize=1)
def _pyarrow() -> tuple[Any, Any] | None:
    """Return the `pyarrow` and `pyarrow.csv` modules, or `None` if not installed."""
    try:
        import pyarrow  # type: ignore[import-untyped]
        import pyarrow.csv  # type: ignore[import-untyped]
    except ImportError:
        return None
    return pyarrow, pyarrow.csv


def _parse_block(
    block: memoryview,
    rows: int,
    columns: int,
    dtype: type[np.float64] | type[np.int64],
) -> NDArray[Any] | None:
    """Parse `rows` lines of `columns` space-separated numbers; `None` if not plain."""
    modules = _pyarrow()
    if modules is None:
        return None
    pa, csv = modules
    allowed = _FLOAT_BYTES if dtype is np.float64 else _INT_BYTES
    if bytes(block).translate(None, allowed):
        return None
    names = [str(i) for i in range(columns)]
    arrow_type = pa.float64() if dtype is np.float64 else pa.int64()
    try:
        table = csv.read_csv(
            pa.py_buffer(block),
            read_options=csv.ReadOptions(column_names=names),
            parse_options=csv.ParseOptions(
                delimiter=" ",
                quote_char=False,
                ignore_empty_lines=False,
            ),
            convert_options=csv.ConvertOptions(
                column_types=dict.fromkeys(names, arrow_type),
                null_values=[],
                strings_can_be_null=False,
            ),
        )
    except pa.ArrowException:
        return None
    if table.num_rows != rows or table.num_columns != columns:
        return None
    result = np.empty((rows, columns), dtype=dtype)
    for i in range(columns):
        column = table.column(i)
        if column.null_count:
            return None
        result[:, i] = column.to_numpy()
    return result


def parse_text_geometry(
    data: bytes,
    point_columns: int,
    record_columns: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int] | None:
    """Parse the two blocks of a text .rec geometry from a byte buffer.

    Args:
        data: file content: point count, points, record count, records, then
            anything (blank space or a metadata appendix).
        point_columns: number of coordinates per point.
        record_columns: number of integers per record (cell indices and labels).

    Returns:
        `(points, records, end)` with `end` the offset of the first byte after the
        last record, or `None` if pyarrow is missing or the buffer is anything but
        plain, well-formed blocks of the declared sizes; the caller then uses the
        general reader.
    """
    view = memoryview(data)
    newlines = np.flatnonzero(np.frombuffer(data, dtype=np.uint8) == 10)
    if newlines.size < 1:
        return None
    try:
        point_count = int(data[: newlines[0]])
        if point_count <= 0 or newlines.size < point_count + 2:
            return None
        record_count = int(data[newlines[point_count] + 1 : newlines[point_count + 1]])
    except ValueError:
        return None
    if record_count <= 0:
        return None
    last = point_count + 1 + record_count
    if newlines.size > last:
        end = int(newlines[last]) + 1
    elif newlines.size == last:
        end = len(data)
    else:
        return None
    points = _parse_block(
        view[newlines[0] + 1 : newlines[point_count] + 1],
        point_count,
        point_columns,
        np.float64,
    )
    if points is None:
        return None
    records = _parse_block(
        view[newlines[point_count + 1] + 1 : end],
        record_count,
        record_columns,
        np.int64,
    )
    if records is None:
        return None
    return points, records, end
