"""Optional trailing metadata appendix for .rec/.arec files.

The appendix is a block of ASCII `# key = value` lines written strictly after the
geometry data (points/topology/labels). Every reader of this format -- text or binary,
2D or 3D -- reads a count and then exactly that many values, so a trailing block is
invisible to it: `_load_text_rec_file`/`_load_2d_text_rec_file` use
`np.loadtxt(f, max_rows=n, ...)`, and the binary readers use
`np.fromfile(f, count=n, dtype=...)`. Both stop reading exactly at the geometry
boundary, so the block must stay trailing -- a leading block would break every reader.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

_APPENDIX_MAGIC = "rec-appendix"
_APPENDIX_VERSION = 1
_APPENDIX_SEQUENCE_INT_KEYS = frozenset({"source_shape", "crop_offset"})
_APPENDIX_SEQUENCE_FLOAT_KEYS = frozenset({"spacing"})
# Bookkeeping keys, always emitted by `write_rec_appendix` itself (the header line and
# the `written_by` line): silently dropped if present in a caller's metadata dict, so
# that feeding a dict from `load_rec_appendix` straight back into `save_rec` doesn't
# duplicate or corrupt them.
_APPENDIX_RESERVED_KEYS = frozenset({"written_by", "_version"})


def _iorec_version() -> str:
    """Return the installed iorec version, or "unknown" if it can't be determined."""
    try:
        return version("iorec")
    except PackageNotFoundError:
        return "unknown"


def _format_appendix_value(value: object) -> str:
    if isinstance(value, (tuple, list, np.ndarray)):
        return " ".join(str(v) for v in value)
    return str(value)


def write_rec_appendix(metadata: dict[str, Any], *, written_by: str) -> bytes:
    """Build the trailing `.rec`/`.arec` metadata appendix block.

    Args:
        metadata: appendix fields, e.g. `coordinate_frame`, `axis_order`, `spacing`,
            `spacing_unit`, `source_shape`, `crop_offset`. Sequence-valued fields
            (`spacing`, `source_shape`, `crop_offset`) are written space-separated.
            Values must be ASCII-only -- `load_rec_appendix` decodes the appendix as
            ASCII. Any of the reserved keys `written_by`/`_version` are ignored, since
            this function always writes its own header and `written_by` line.
        written_by: a version string stamped on the appendix by the writer.

    Raises:
        UnicodeEncodeError: if a metadata value contains a non-ASCII character.

    Returns:
        bytes: the ASCII appendix block (including a trailing newline), ready to be
        appended after the geometry bytes/text already written.
    """
    lines = [f"# {_APPENDIX_MAGIC} {_APPENDIX_VERSION}", f"# written_by = {written_by}"]
    for key, raw_value in metadata.items():
        if key in _APPENDIX_RESERVED_KEYS:
            continue
        lines.append(f"# {key} = {_format_appendix_value(raw_value)}")
    return ("\n".join(lines) + "\n").encode("ascii")


def parse_rec_appendix(block_text: str) -> dict[str, Any]:
    """Parse a `.rec`/`.arec` metadata appendix block into a dict.

    Args:
        block_text: the appendix block: one `# key = value` line per line, headed by
            the `# rec-appendix <version>` line.

    Raises:
        ValueError: if `block_text` does not start with a `rec-appendix <version>`
            header.

    Returns:
        dict[str, Any]: parsed fields, plus `"_version"` (the schema version, an int).
        Sequence-valued fields known to the schema (`spacing`, `source_shape`,
        `crop_offset`) are parsed into tuples of `float`/`int`; unknown keys are kept as
        raw strings, never dropped, so a future rewrite can carry them forward.
    """
    lines = [ln.strip() for ln in block_text.splitlines() if ln.strip()]
    header = lines[0] if lines else ""
    header_body = header[1:].strip() if header.startswith("#") else ""
    header_parts = header_body.split()
    if len(header_parts) != 2 or header_parts[0] != _APPENDIX_MAGIC:
        error = f"Not a rec-appendix block, expected a header line, got: {header!r}"
        raise ValueError(error)

    metadata: dict[str, Any] = {"_version": int(header_parts[1])}
    for line in lines[1:]:
        content = line[1:].strip() if line.startswith("#") else line
        key, _, raw_value = content.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        if key in _APPENDIX_SEQUENCE_INT_KEYS:
            metadata[key] = tuple(int(v) for v in raw_value.split())
        elif key in _APPENDIX_SEQUENCE_FLOAT_KEYS:
            metadata[key] = tuple(float(v) for v in raw_value.split())
        else:
            metadata[key] = raw_value
    return metadata


def _appendix_marker() -> bytes:
    return f"# {_APPENDIX_MAGIC} ".encode("ascii")


def load_rec_appendix(filename: str | Path) -> dict[str, Any] | None:
    """Read the trailing metadata appendix from a `.rec`/`.arec` file, if present.

    Works uniformly for text- and binary-mode, 2D and 3D files: the appendix is located
    by finding the last `# rec-appendix ` marker in the file. It is written only ever as
    the literal first line of the trailing appendix block (see `write_rec_appendix`), so
    its last occurrence marks the start of the block -- note that binary-mode payloads
    have no guaranteed newline before this marker, so the search cannot require one.

    Args:
        filename (str | Path): the path to the file on disk.

    Raises:
        FileNotFoundError: if the file doesn't exist.

    Returns:
        dict[str, Any] | None: the parsed appendix fields (see `parse_rec_appendix`), or
        `None` if the file has no appendix. A `None` return must be treated by the
        caller as "no information at all" -- e.g. never as isotropic/unit spacing --
        since every file saved before this feature existed has no appendix.
    """
    filename = Path(filename).resolve()
    if not filename.exists():
        error = f"{filename} not found."
        raise FileNotFoundError(error)
    raw = filename.read_bytes()
    idx = raw.rfind(_appendix_marker())
    if idx == -1:
        return None
    return parse_rec_appendix(raw[idx:].decode("ascii"))


def _split_rec_bytes(raw: bytes) -> tuple[bytes, dict[str, Any] | None]:
    """Split raw `.rec`/`.arec` file bytes into geometry bytes and parsed metadata.

    Lets a caller that already has the whole file in memory get both geometry and
    metadata from that single buffer, instead of reading the file a second time (as
    calling `load_rec_appendix` separately would) just to look for the appendix.

    Args:
        raw: the full raw bytes of a `.rec`/`.arec` file.

    Returns:
        tuple[bytes, dict[str, Any] | None]: the geometry bytes (everything before the
        appendix, or the whole input if there is none) and the parsed metadata (`None`
        if there is no appendix) -- see `load_rec_appendix`.
    """
    idx = raw.rfind(_appendix_marker())
    if idx == -1:
        return raw, None
    return raw[:idx], parse_rec_appendix(raw[idx:].decode("ascii"))
