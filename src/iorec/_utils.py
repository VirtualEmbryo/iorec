"""Shared helpers for .rec/.arec files, used by both the 2D and 3D readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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
