# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2023-2026 the iorec authors
"""Optional trailing metadata appendix for .rec/.arec files.

The appendix is a block of ASCII `#` lines written strictly after the geometry data
(points/topology/labels). Every reader of this format -- text or binary, 2D or 3D --
reads a count and then exactly that many values, so a trailing block is invisible to
it: `_load_text_rec_file`/`_load_2d_text_rec_file` use `np.loadtxt(f, max_rows=n, ...)`,
and the binary readers use `np.fromfile(f, count=n, dtype=...)`. Both stop reading
exactly at the geometry boundary, so the block must stay trailing -- a leading block
would break every reader.

Version 1 writes `# key = value` lines and types only `spacing`, `source_shape` and
`crop_offset`; version 2 writes `# key: <JSON>` lines, so every value carries its type.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import GenericAlias
from typing import Any

import numpy as np

_APPENDIX_MAGIC = "rec-appendix"
# Version written when the caller does not choose one.
_DEFAULT_APPENDIX_VERSION = 2
_APPENDIX_SEQUENCE_INT_KEYS = frozenset({"source_shape", "crop_offset"})
_APPENDIX_SEQUENCE_FLOAT_KEYS = frozenset({"spacing"})
# Bookkeeping keys, always emitted by `write_rec_appendix` itself (the header line, the
# `written_by` line, and the optional `writer_library` line): silently dropped if
# present in a caller's metadata dict, so that feeding a dict from
# `load_rec_appendix` straight back into `save_rec` doesn't duplicate or corrupt them.
_APPENDIX_RESERVED_KEYS = frozenset({"written_by", "writer_library", "_version"})
# Characters a version-2 key may not contain: the key/value separator, the comment
# sign that starts every appendix line, and whitespace.
_FORBIDDEN_KEY_CHARACTERS = frozenset(":#") | frozenset(" \t\r\n\f\v")
_TRUE_TEXTS = frozenset({"True", "true", "1"})
_FALSE_TEXTS = frozenset({"False", "false", "0"})


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


def _json_scalar(value: object, where: str) -> bool | int | float | str | None:
    """Convert a scalar metadata value to a JSON scalar, or return `None` if not one."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            error = (
                f"Metadata {where} is {number!r}: appendix version 2 stores JSON, "
                "which has no NaN or infinity."
            )
            raise ValueError(error)
        return number
    if isinstance(value, str):
        return value
    return None


def _json_compatible(value: object, where: str) -> object:
    """Convert a metadata value to plain JSON types, refusing what JSON cannot hold."""
    scalar = _json_scalar(value, where)
    if scalar is not None:
        return scalar
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist(), where)
    if isinstance(value, (list, tuple)):
        return [_json_compatible(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                error = f"Metadata {where} has a non-string key {key!r}."
                raise TypeError(error)
            converted[key] = _json_compatible(item, f"{where}[{key!r}]")
        return converted
    error = (
        f"Metadata {where} has type {type(value).__name__}: appendix version 2 "
        "stores numbers, strings, booleans, sequences and mappings only."
    )
    raise TypeError(error)


def _check_version_2_key(key: object) -> str:
    if (
        not isinstance(key, str)
        or not key
        or not key.isascii()
        or any(c in _FORBIDDEN_KEY_CHARACTERS for c in key)
    ):
        error = (
            f"Metadata key {key!r} cannot be written in appendix version 2: a key is "
            "a non-empty ASCII string without whitespace, ':' or '#'."
        )
        raise ValueError(error)
    return key


def _json_text(value: object, where: str) -> str:
    # `ensure_ascii` keeps the block ASCII; escaping `#` (legal in JSON strings only)
    # keeps the `# rec-appendix ` marker out of every value, so the reader's search for
    # its last occurrence always finds the header.
    text = json.dumps(
        _json_compatible(value, where),
        ensure_ascii=True,
        allow_nan=False,
    )
    return text.replace("#", "\\u0023")


def write_rec_appendix(
    metadata: dict[str, Any],
    *,
    written_by: str,
    writer_library: str | None = None,
    appendix_version: int = _DEFAULT_APPENDIX_VERSION,
) -> bytes:
    """Build the trailing `.rec`/`.arec` metadata appendix block.

    Args:
        metadata: appendix fields, e.g. `coordinate_frame`, `axis_order`, `spacing`,
            `spacing_unit`, `source_shape`, `crop_offset`. In version 1, values are
            written with `str` and sequences space-separated; values must be ASCII.
            In version 2, each value is written as JSON: numbers, strings, booleans,
            sequences and mappings with string keys, nested at will; any string.
            The reserved keys `written_by`/`writer_library`/`_version` are ignored,
            since this function always writes those itself.
        written_by: a version string stamped on the appendix by the writer.
        writer_library: an optional `library version` string stamped on its own line
            right after `written_by`, identifying the library that serialised the
            file. Lets both facts survive when `written_by` names something else (the
            program that produced the mesh): what produced it, and what wrote the
            bytes. Omitted entirely when `None`.
        appendix_version: 1 or 2. Defaults to 2.

    Raises:
        UnicodeEncodeError: in version 1, if a value contains a non-ASCII character.
        ValueError: if `appendix_version` is not 1 or 2; in version 2, for a key that
            is empty, non-ASCII or contains whitespace, ':' or '#', or for a NaN or
            infinite number.
        TypeError: in version 2, for a value JSON cannot hold (e.g. `None`).

    Returns:
        bytes: the ASCII appendix block (including a trailing newline), ready to be
        appended after the geometry bytes/text already written.
    """
    if appendix_version == 1:
        lines = [f"# {_APPENDIX_MAGIC} 1", f"# written_by = {written_by}"]
        if writer_library is not None:
            lines.append(f"# writer_library = {writer_library}")
        for key, raw_value in metadata.items():
            if key in _APPENDIX_RESERVED_KEYS:
                continue
            lines.append(f"# {key} = {_format_appendix_value(raw_value)}")
        return ("\n".join(lines) + "\n").encode("ascii")
    if appendix_version == 2:
        lines = [
            f"# {_APPENDIX_MAGIC} 2",
            f"# written_by: {_json_text(written_by, 'written_by')}",
        ]
        if writer_library is not None:
            text = _json_text(writer_library, "writer_library")
            lines.append(f"# writer_library: {text}")
        for key, raw_value in metadata.items():
            if key in _APPENDIX_RESERVED_KEYS:
                continue
            checked = _check_version_2_key(key)
            lines.append(f"# {checked}: {_json_text(raw_value, repr(checked))}")
        return ("\n".join(lines) + "\n").encode("ascii")
    error = f"appendix_version must be 1 or 2, got {appendix_version!r}."
    raise ValueError(error)


def _typed_version_1_value(key: str, raw_value: str, kind: object) -> object:
    """Convert the text of a version-1 value to the type the caller asked for."""
    if kind is tuple or kind is list:
        error = f"Metadata type for key {key!r}: use tuple[X, ...], not {kind!r}."
        raise TypeError(error)
    try:
        if kind is bool:
            if raw_value in _TRUE_TEXTS:
                return True
            if raw_value in _FALSE_TEXTS:
                return False
            raise ValueError(raw_value)
        if isinstance(kind, GenericAlias) and kind.__origin__ is tuple:
            item_kind, *rest = kind.__args__
            if rest != [Ellipsis]:
                error = f"Metadata type for key {key!r}: use tuple[X, ...], not {kind}."
                raise TypeError(error)
            return tuple(
                _typed_version_1_value(key, v, item_kind) for v in raw_value.split()
            )
        if callable(kind):
            return kind(raw_value)
    except ValueError as exc:
        error = f"Metadata key {key!r}: cannot read {raw_value!r} as {kind}."
        raise ValueError(error) from exc
    error = f"Metadata type for key {key!r} is not callable: {kind!r}."
    raise TypeError(error)


def _parse_version_1_lines(
    lines: list[str],
    metadata: dict[str, Any],
    metadata_types: Mapping[str, object] | None,
) -> None:
    for line in lines:
        content = line[1:].strip() if line.startswith("#") else line
        key, _, raw_value = content.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        if metadata_types is not None and key in metadata_types:
            metadata[key] = _typed_version_1_value(key, raw_value, metadata_types[key])
        elif key in _APPENDIX_SEQUENCE_INT_KEYS:
            metadata[key] = tuple(int(v) for v in raw_value.split())
        elif key in _APPENDIX_SEQUENCE_FLOAT_KEYS:
            metadata[key] = tuple(float(v) for v in raw_value.split())
        else:
            metadata[key] = raw_value


def _refuse_constant(name: str) -> None:
    error = f"{name} is not a JSON number"
    raise ValueError(error)


def _tuples_for_arrays(value: object, key: str) -> object:
    if value is None:
        error = f"Metadata key {key!r} holds a JSON null, which version 2 excludes."
        raise ValueError(error)
    if isinstance(value, list):
        return tuple(_tuples_for_arrays(v, key) for v in value)
    if isinstance(value, dict):
        return {k: _tuples_for_arrays(v, key) for k, v in value.items()}
    return value


def _parse_version_2_lines(lines: list[str], metadata: dict[str, Any]) -> None:
    for line in lines:
        content = line[1:].strip() if line.startswith("#") else line
        key, separator, raw_value = content.partition(":")
        key = key.strip()
        if not separator or not key:
            error = f"Appendix version 2 line without a 'key: value' form: {line!r}"
            raise ValueError(error)
        try:
            value = json.loads(raw_value, parse_constant=_refuse_constant)
        except ValueError as exc:
            error = f"Metadata key {key!r}: {raw_value.strip()!r} is not valid JSON."
            raise ValueError(error) from exc
        metadata[key] = _tuples_for_arrays(value, key)


def _parse_rec_appendix(
    block_text: str,
    metadata_types: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Parse a `.rec`/`.arec` metadata appendix block into a dict.

    Args:
        block_text: the appendix block, headed by the `# rec-appendix <version>` line.
        metadata_types: for version-1 blocks only, a type per key (see
            `load_rec_appendix`); ignored for version 2, whose values carry their type.

    Raises:
        ValueError: if `block_text` does not start with a `rec-appendix <version>`
            header, if the version is not 1 or 2, or if a value cannot be read.

    Returns:
        dict[str, Any]: parsed fields, plus `"_version"` (the schema version, an int).
        Version 1: `spacing`, `source_shape` and `crop_offset` are tuples of
        `float`/`int`, keys named in `metadata_types` have that type, other keys are
        kept as raw strings, never dropped. Version 2: JSON values, arrays as tuples.
    """
    lines = [ln.strip() for ln in block_text.splitlines() if ln.strip()]
    header = lines[0] if lines else ""
    header_body = header[1:].strip() if header.startswith("#") else ""
    header_parts = header_body.split()
    if len(header_parts) != 2 or header_parts[0] != _APPENDIX_MAGIC:
        error = f"Not a rec-appendix block, expected a header line, got: {header!r}"
        raise ValueError(error)

    appendix_version = int(header_parts[1])
    metadata: dict[str, Any] = {"_version": appendix_version}
    if appendix_version == 1:
        _parse_version_1_lines(lines[1:], metadata, metadata_types)
    elif appendix_version == 2:
        _parse_version_2_lines(lines[1:], metadata)
    else:
        error = (
            f"Appendix version {appendix_version} is not supported by iorec "
            f"{_iorec_version()}, which reads versions 1 and 2."
        )
        raise ValueError(error)
    return metadata


def _appendix_marker() -> bytes:
    return f"# {_APPENDIX_MAGIC} ".encode("ascii")


def load_rec_appendix(
    filename: str | Path,
    *,
    metadata_types: Mapping[str, object] | None = None,
) -> dict[str, Any] | None:
    """Read the trailing metadata appendix from a `.rec`/`.arec` file, if present.

    Works uniformly for text- and binary-mode, 2D and 3D files: the appendix is located
    by finding the last `# rec-appendix ` marker in the file. It is written only ever as
    the literal first line of the trailing appendix block (see `write_rec_appendix`), so
    its last occurrence marks the start of the block -- note that binary-mode payloads
    have no guaranteed newline before this marker, so the search cannot require one.

    Args:
        filename (str | Path): the path to the file on disk.
        metadata_types: version-1 files only: the type of the value of each named key,
            one of `str`, `int`, `float`, `bool`, `tuple[X, ...]` (space-separated
            items of type X), or any function of the value's text. Keys absent from
            the file are ignored; version-2 files are returned as stored.

    Raises:
        FileNotFoundError: if the file doesn't exist.
        ValueError: if the appendix is malformed, of an unknown version, or a value
            cannot be read with the type asked for.

    Returns:
        dict[str, Any] | None: the parsed appendix fields (see `_parse_rec_appendix`),
        or `None` if the file has no appendix. A `None` return must be treated by the
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
    return _parse_rec_appendix(raw[idx:].decode("ascii"), metadata_types)


def _split_rec_bytes(
    raw: bytes,
    metadata_types: Mapping[str, object] | None = None,
) -> tuple[bytes, dict[str, Any] | None]:
    """Split raw `.rec`/`.arec` file bytes into geometry bytes and parsed metadata.

    Lets a caller that already has the whole file in memory get both geometry and
    metadata from that single buffer, instead of reading the file a second time (as
    calling `load_rec_appendix` separately would) just to look for the appendix.

    Args:
        raw: the full raw bytes of a `.rec`/`.arec` file.
        metadata_types: as in `load_rec_appendix`.

    Returns:
        tuple[bytes, dict[str, Any] | None]: the geometry bytes (everything before the
        appendix, or the whole input if there is none) and the parsed metadata (`None`
        if there is no appendix) -- see `load_rec_appendix`.
    """
    idx = raw.rfind(_appendix_marker())
    if idx == -1:
        return raw, None
    return raw[:idx], _parse_rec_appendix(raw[idx:].decode("ascii"), metadata_types)
