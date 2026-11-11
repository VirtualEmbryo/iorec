"""Typed values in the .rec/.arec metadata appendix: version 2, and typed version-1 reads."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import iorec
from iorec.rec_appendix import (
    _parse_rec_appendix,
    _parse_version_1_lines,
    write_rec_appendix,
)

# One value of every kind version 2 stores, with the value and exact type read back.
EVERY_VALUE_TYPE: dict[str, tuple[Any, Any]] = {
    "integer": (2048, 2048),
    "negative_integer": (-3, -3),
    "large_integer": (2**62 + 1, 2**62 + 1),
    "float": (3.5, 3.5),
    "float_without_fraction": (1.0, 1.0),
    "float_shortest_repr": (0.1, 0.1),
    "tiny_float": (5e-324, 5e-324),
    "true": (True, True),
    "false": (False, False),
    "string": ("voxel_index", "voxel_index"),
    "empty_string": ("", ""),
    "unicode_string": ("µm, é, 細胞", "µm, é, 細胞"),
    "string_with_separators": ("a = b: c # d", "a = b: c # d"),
    "string_with_marker": ("# rec-appendix 1\n# x = y", "# rec-appendix 1\n# x = y"),
    "empty_list": ([], ()),
    "list_of_integers": ([0, 7], (0, 7)),
    "tuple_of_floats": ((1.0, 0.325, 0.325), (1.0, 0.325, 0.325)),
    "list_of_strings": (["z_min", "y_max"], ("z_min", "y_max")),
    "mixed_list": ([1, 2.5, "a", True], (1, 2.5, "a", True)),
    "nested_lists": ([[1, 2], [3], []], ((1, 2), (3,), ())),
    "mapping": ({"3": 112, "9": 4}, {"3": 112, "9": 4}),
    "empty_mapping": ({}, {}),
    "nested_mapping": (
        {"a": {"b": [1, {"c": False}]}},
        {"a": {"b": (1, {"c": False})}},
    ),
    "numpy_integer": (np.int64(7), 7),
    "numpy_float32": (np.float32(0.1), float(np.float32(0.1))),
    "numpy_bool": (np.bool_(True), True),
    "numpy_array_1d": (np.array([194, 200, 199]), (194, 200, 199)),
    "numpy_array_2d": (np.array([[0.5, 1.0], [2.0, 4.0]]), ((0.5, 1.0), (2.0, 4.0))),
}


def _assert_same_value_and_type(read: object, expected: object) -> None:
    """Compare recursively, requiring the same type at every level (1 is not 1.0)."""
    assert type(read) is type(expected), (read, expected)
    if isinstance(expected, tuple):
        assert isinstance(read, tuple)
        assert len(read) == len(expected)
        for r, e in zip(read, expected, strict=True):
            _assert_same_value_and_type(r, e)
    elif isinstance(expected, dict):
        assert isinstance(read, dict)
        assert read.keys() == expected.keys()
        for key in expected:
            _assert_same_value_and_type(read[key], expected[key])
    else:
        assert read == expected


def _mesh(dimension: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dimension == 3:
        return iorec.load_rec("tests/assets/text.rec")
    return iorec.load_2d_rec("tests/assets/text_2d.rec")


def _save(dimension: int, filename: Path, binary_mode: bool, **options: Any) -> None:  # noqa: ANN401
    save = iorec.save_rec if dimension == 3 else iorec.save_2d_rec
    save(filename, *_mesh(dimension), binary_mode, **options)


def _load_with_metadata(dimension: int, filename: Path, **options: Any) -> tuple:  # noqa: ANN401
    if dimension == 3:
        return iorec.load_rec_with_metadata(filename, **options)
    return iorec.load_2d_rec_with_metadata(filename, **options)


DIMENSIONS_AND_ENCODINGS = pytest.mark.parametrize(
    ("dimension", "binary_mode"),
    [(2, False), (2, True), (3, False), (3, True)],
)


@DIMENSIONS_AND_ENCODINGS
def test_version_2_round_trips_every_value_type(
    tmp_path: Path,
    dimension: int,
    binary_mode: bool,
) -> None:
    """Every value kind reads back with its value and type; arrays come back as tuples."""
    filename = tmp_path / "mesh.rec"
    metadata = {key: value for key, (value, _) in EVERY_VALUE_TYPE.items()}
    _save(dimension, filename, binary_mode, metadata=metadata, appendix_version=2)

    *mesh, read = _load_with_metadata(dimension, filename, validate=True)

    for array, expected in zip(mesh, _mesh(dimension), strict=True):
        assert np.array_equal(array, expected)
    assert read == iorec.load_rec_appendix(filename)
    assert read["_version"] == 2
    assert set(read) == {"_version", "written_by", *EVERY_VALUE_TYPE}
    for key, (_, expected) in EVERY_VALUE_TYPE.items():
        _assert_same_value_and_type(read[key], expected)


@DIMENSIONS_AND_ENCODINGS
def test_version_1_keeps_its_types_and_reads_caller_types(
    tmp_path: Path,
    dimension: int,
    binary_mode: bool,
) -> None:
    """Version 1: frame keys typed, others strings unless the caller names a type."""
    filename = tmp_path / "mesh.rec"
    metadata = {
        "spacing": (1.0, 0.325, 0.325),
        "source_shape": (194, 200, 199),
        "media": [0, 7],
        "crop_faces": ("z_min", "y_max"),
        "enclosed_media_voxels": 2048,
        "enclosed_media_inscribed_radius": 3.5,
        "resampled_spacing": np.array([0.5, 0.5]),
        "flag": True,
        "note": "kept as text",
    }
    _save(dimension, filename, binary_mode, metadata=metadata, appendix_version=1)
    types = {
        "media": tuple[int, ...],
        "crop_faces": tuple[str, ...],
        "enclosed_media_voxels": int,
        "enclosed_media_inscribed_radius": float,
        "resampled_spacing": tuple[float, ...],
        "flag": bool,
        "note": str.upper,
        "absent_key": int,
    }

    untyped = iorec.load_rec_appendix(filename)
    typed = iorec.load_rec_appendix(filename, metadata_types=types)
    *_, typed_with_mesh = _load_with_metadata(dimension, filename, metadata_types=types)

    assert untyped is not None
    assert typed == typed_with_mesh
    _assert_same_value_and_type(untyped["spacing"], (1.0, 0.325, 0.325))
    _assert_same_value_and_type(untyped["source_shape"], (194, 200, 199))
    assert untyped["media"] == "0 7"
    assert untyped["flag"] == "True"
    assert untyped["enclosed_media_voxels"] == "2048"
    expected = {
        "_version": 1,
        "written_by": untyped["written_by"],
        "spacing": (1.0, 0.325, 0.325),
        "source_shape": (194, 200, 199),
        "media": (0, 7),
        "crop_faces": ("z_min", "y_max"),
        "enclosed_media_voxels": 2048,
        "enclosed_media_inscribed_radius": 3.5,
        "resampled_spacing": (0.5, 0.5),
        "flag": True,
        "note": "KEPT AS TEXT",
    }
    _assert_same_value_and_type(typed, expected)


def test_version_1_without_types_is_unchanged() -> None:
    """A version-1 block reads to the same values and types with or without a map."""
    block = (
        "# rec-appendix 1\n# written_by = 1.4.0\n# spacing = 1.0 0.5\n"
        "# source_shape = 3 4\n# crop_offset = 0 1\n# media = 0 7\n"
    )
    expected = {
        "_version": 1,
        "written_by": "1.4.0",
        "spacing": (1.0, 0.5),
        "source_shape": (3, 4),
        "crop_offset": (0, 1),
        "media": "0 7",
    }

    _assert_same_value_and_type(_parse_rec_appendix(block), expected)
    _assert_same_value_and_type(_parse_rec_appendix(block, {}), expected)
    _assert_same_value_and_type(_parse_rec_appendix(block, {"other": int}), expected)


def test_types_are_ignored_for_version_2(tmp_path: Path) -> None:
    """Version-2 values carry their own type; a caller's type map does not apply."""
    filename = tmp_path / "mesh.rec"
    _save(3, filename, False, metadata={"media": [0, 7]}, appendix_version=2)

    read = iorec.load_rec_appendix(filename, metadata_types={"media": str})

    assert read is not None
    assert read["media"] == (0, 7)


@pytest.mark.parametrize(
    ("raw", "kind"),
    [("maybe", bool), ("1.5", int), ("a b", tuple[float, ...]), ("x", float)],
)
def test_version_1_type_error_names_the_key(raw: str, kind: object) -> None:
    """A value that does not read as the type asked for raises, naming the key."""
    block = f"# rec-appendix 1\n# written_by = 1.4.0\n# media = {raw}\n"

    with pytest.raises(ValueError, match="'media'"):
        _parse_rec_appendix(block, {"media": kind})


@pytest.mark.parametrize("kind", [tuple, list, tuple[int, int], 3])
def test_version_1_unusable_type_is_refused(kind: object) -> None:
    """Bare `tuple`, fixed-length tuples and non-callables are refused, not guessed."""
    block = "# rec-appendix 1\n# written_by = 1.4.0\n# media = 0 7\n"

    with pytest.raises(TypeError, match="'media'"):
        _parse_rec_appendix(block, {"media": kind})


def test_default_version_is_2(tmp_path: Path) -> None:
    """Without `appendix_version`, the writers write version 2."""
    filename = tmp_path / "mesh.rec"
    _save(3, filename, False, metadata={"media": [0, 7]})

    assert b"# rec-appendix 2\n" in filename.read_bytes()
    assert iorec.load_rec_appendix(filename) == {
        "_version": 2,
        "written_by": iorec.load_rec_appendix(filename)["written_by"],
        "media": (0, 7),
    }


def test_version_2_pins_wire_format() -> None:
    """The version-2 block has an exact byte layout."""
    metadata = {
        "coordinate_frame": "physical",
        "spacing": (1.0, 0.5),
        "media": [0, 7],
        "flag": True,
        "voxels": {"3": 112},
        "note": "# é",
    }

    block = write_rec_appendix(
        metadata,
        written_by="my-pipeline 2.1.0",
        writer_library="iorec 1.5.0",
        appendix_version=2,
    )

    expected = (
        b"# rec-appendix 2\n"
        b'# written_by: "my-pipeline 2.1.0"\n'
        b'# writer_library: "iorec 1.5.0"\n'
        b'# coordinate_frame: "physical"\n'
        b"# spacing: [1.0, 0.5]\n"
        b"# media: [0, 7]\n"
        b"# flag: true\n"
        b'# voxels: {"3": 112}\n'
        b'# note: "\\u0023 \\u00e9"\n'
    )
    assert block == expected


def test_version_2_block_does_not_break_the_version_1_line_parser() -> None:
    """Version-1 readers (iorec <= 1.4.0) parse version-2 lines without raising.

    `_parse_version_1_lines` is the line parser of iorec 1.1.0 to 1.4.0: with `key: JSON`
    lines, no line has a frame key before an `=`, so it never converts JSON to numbers.
    """
    metadata = {
        "spacing": [1.0, 0.5, 0.5],
        "source_shape": [3, 4, 5],
        "crop_offset": [0, 1, 2],
        "note": "spacing = 1 x",
    }
    block = write_rec_appendix(metadata, written_by="test", appendix_version=2)
    lines = [ln.strip() for ln in block.decode("ascii").splitlines()][1:]
    parsed: dict[str, Any] = {}

    _parse_version_1_lines(lines, parsed, None)

    assert not {"spacing", "source_shape", "crop_offset"} & parsed.keys()


@pytest.mark.parametrize(
    "value",
    [
        math.nan,
        math.inf,
        -math.inf,
        np.float64("nan"),
        [1.0, math.inf],
        {"a": [math.nan]},
        np.array([0.5, np.nan]),
    ],
)
@pytest.mark.parametrize("dimension", [2, 3])
def test_version_2_refuses_non_finite_numbers(
    tmp_path: Path,
    value: object,
    dimension: int,
) -> None:
    """NaN and infinities raise a clear error, and no file is written."""
    filename = tmp_path / "mesh.rec"

    with pytest.raises(ValueError, match="no NaN or infinity"):
        _save(dimension, filename, False, metadata={"x": value}, appendix_version=2)

    assert not filename.exists()


@pytest.mark.parametrize(
    "value",
    [None, object(), {1: "a"}, b"bytes", {1, 2}, 1 + 2j, Path("a")],
)
def test_version_2_refuses_values_json_cannot_hold(value: object) -> None:
    """None, bytes, sets, complex numbers, paths and non-string keys raise TypeError."""
    with pytest.raises(TypeError, match="Metadata"):
        write_rec_appendix({"x": value}, written_by="test", appendix_version=2)


@pytest.mark.parametrize("key", ["", "a b", "a:b", "a#b", "é", "a\tb"])
def test_version_2_refuses_keys_it_cannot_write(key: str) -> None:
    """Keys must be non-empty ASCII without whitespace, ':' or '#'."""
    with pytest.raises(ValueError, match="cannot be written in appendix version 2"):
        write_rec_appendix({key: 1}, written_by="test", appendix_version=2)


@pytest.mark.parametrize("version", [0, 3])
def test_unknown_version_is_refused_on_write_and_read(version: int) -> None:
    """Only versions 1 and 2 are written or read."""
    with pytest.raises(ValueError, match="appendix_version must be 1 or 2"):
        write_rec_appendix({}, written_by="test", appendix_version=version)
    with pytest.raises(ValueError, match="reads versions 1 and 2"):
        _parse_rec_appendix(f"# rec-appendix {version}\n# a: 1\n")


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("# x: NaN", "not valid JSON"),
        ("# x: -Infinity", "not valid JSON"),
        ("# x: [1, null]", "null"),
        ("# x: [1,", "not valid JSON"),
        ("# x = 1", "'key: value'"),
    ],
)
def test_version_2_reader_refuses_malformed_lines(line: str, message: str) -> None:
    """A hand-edited version-2 block with non-JSON values raises, naming the problem."""
    with pytest.raises(ValueError, match=message):
        _parse_rec_appendix(f"# rec-appendix 2\n{line}\n")


@pytest.mark.parametrize("version", [1, 2])
def test_convert_keeps_the_appendix_version(tmp_path: Path, version: int) -> None:
    """`convert` writes the appendix in the version of its source."""
    source = tmp_path / "m.rec"
    _save(3, source, False, metadata={"media": [0, 7]}, appendix_version=version)

    iorec.convert(source, tmp_path / "n.arec", save_options={"binary_mode": True})

    copied = iorec.load_rec_appendix(tmp_path / "n.arec")
    assert copied is not None
    assert copied["_version"] == version
    assert copied["media"] == ("0 7" if version == 1 else (0, 7))
