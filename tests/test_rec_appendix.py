"""Test file for the .rec/.arec metadata appendix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import iorec
from iorec.rec_appendix import _parse_rec_appendix, write_rec_appendix


def test_write_parse_round_trip() -> None:
    """write_rec_appendix/_parse_rec_appendix round trip all field kinds."""
    metadata = {
        "coordinate_frame": "voxel_index",
        "source_shape": (194, 200, 199),
        "crop_offset": (0, 0, 0),
        "spacing": (1.0, 0.325, 0.325),
        "some_unknown_field": "kept-as-string",
    }

    block = write_rec_appendix(metadata, written_by="test 0.0.0")
    parsed = _parse_rec_appendix(block.decode("ascii"))

    assert parsed["_version"] == 1
    assert parsed["written_by"] == "test 0.0.0"
    assert parsed["coordinate_frame"] == "voxel_index"
    assert parsed["source_shape"] == (194, 200, 199)
    assert parsed["crop_offset"] == (0, 0, 0)
    assert parsed["spacing"] == (1.0, 0.325, 0.325)
    assert parsed["some_unknown_field"] == "kept-as-string"


def test_write_rec_appendix_pins_wire_format() -> None:
    """The wire format is pinned to an exact byte layout.

    Independent of the installed package version.
    """
    metadata = {"coordinate_frame": "physical", "spacing": (1.0, 0.5)}

    block = write_rec_appendix(metadata, written_by="test 0.0.0")

    expected = (
        b"# rec-appendix 1\n"
        b"# written_by = test 0.0.0\n"
        b"# coordinate_frame = physical\n"
        b"# spacing = 1.0 0.5\n"
    )
    assert block == expected


def test_write_rec_appendix_drops_reserved_keys() -> None:
    """`written_by`/`_version` in the metadata dict are ignored, not duplicated."""
    metadata = {"_version": 999, "written_by": "someone-else", "spacing_unit": "um"}

    block = write_rec_appendix(metadata, written_by="test 0.0.0")

    assert block.count(b"written_by") == 1
    assert b"_version" not in block
    parsed = _parse_rec_appendix(block.decode("ascii"))
    assert parsed["_version"] == 1
    assert parsed["written_by"] == "test 0.0.0"
    assert parsed["spacing_unit"] == "um"


def test_write_rec_appendix_rejects_non_ascii() -> None:
    """A non-ASCII metadata value raises before any file is written."""
    with pytest.raises(UnicodeEncodeError):
        write_rec_appendix({"spacing_unit": "µm"}, written_by="test 0.0.0")


def _mesh_3d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, triangles, labels = iorec.load_rec("tests/assets/text.rec")
    return points, triangles, labels


def _mesh_2d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points, segments, labels = iorec.load_2d_rec("tests/assets/text_2d.rec")
    return points, segments, labels


@pytest.mark.parametrize("binary_mode", [False, True])
def test_save_rec_metadata_round_trip_3d(tmp_path: Path, binary_mode: bool) -> None:
    """save_rec(metadata=...) round trips through load_rec_appendix, in 3D."""
    points, triangles, labels = _mesh_3d()
    filename = tmp_path / "mesh.rec"
    metadata = {"coordinate_frame": "voxel_index", "spacing": (1.0, 0.325, 0.325)}

    iorec.save_rec(
        filename,
        points,
        triangles,
        labels,
        binary_mode=binary_mode,
        metadata=metadata,
    )

    saved_points, saved_triangles, saved_labels = iorec.load_rec(filename)
    assert np.array_equal(points, saved_points)
    assert np.array_equal(triangles, saved_triangles)
    assert np.array_equal(labels, saved_labels)

    appendix = iorec.load_rec_appendix(filename)
    assert appendix is not None
    assert appendix["coordinate_frame"] == "voxel_index"
    assert appendix["spacing"] == (1.0, 0.325, 0.325)


@pytest.mark.parametrize("binary_mode", [False, True])
def test_save_2d_rec_metadata_round_trip(tmp_path: Path, binary_mode: bool) -> None:
    """save_2d_rec(metadata=...) round trips through load_rec_appendix, in 2D.

    In particular, the sequence-valued fields (spacing, source_shape, crop_offset) are
    parsed generically -- no hardcoded 3-tuple arity -- so 2-tuples round trip as
    2-tuples, matching a 2D file's dimensionality, not the 3D shape shown in the
    README example.
    """
    points, segments, labels = _mesh_2d()
    filename = tmp_path / "mesh2d.rec"
    metadata = {
        "coordinate_frame": "physical",
        "spacing": (0.5, 0.5),
        "source_shape": (200, 200),
        "crop_offset": (0, 0),
    }

    iorec.save_2d_rec(
        filename,
        points,
        segments,
        labels,
        binary_mode=binary_mode,
        metadata=metadata,
    )

    saved_points, saved_segments, saved_labels = iorec.load_2d_rec(filename)
    assert np.array_equal(points, saved_points)
    assert np.array_equal(segments, saved_segments)
    assert np.array_equal(labels, saved_labels)

    appendix = iorec.load_rec_appendix(filename)
    assert appendix is not None
    assert appendix["coordinate_frame"] == "physical"
    assert appendix["spacing"] == (0.5, 0.5)
    assert appendix["source_shape"] == (200, 200)
    assert appendix["crop_offset"] == (0, 0)
    assert all(isinstance(v, float) for v in appendix["spacing"])
    assert all(isinstance(v, int) for v in appendix["source_shape"])
    assert all(isinstance(v, int) for v in appendix["crop_offset"])


def test_load_rec_appendix_none_when_absent() -> None:
    """A file saved without metadata has no appendix."""
    assert iorec.load_rec_appendix("tests/assets/text.rec") is None


def test_load_rec_appendix_empty_dict_is_not_none(tmp_path: Path) -> None:
    """An explicitly empty metadata dict is not the same as no metadata at all."""
    points, triangles, labels = _mesh_3d()
    filename = tmp_path / "mesh.rec"

    iorec.save_rec(filename, points, triangles, labels, metadata={})

    appendix = iorec.load_rec_appendix(filename)
    assert appendix is not None
    assert set(appendix.keys()) == {"_version", "written_by"}


def test_save_rec_metadata_error_writes_no_file(tmp_path: Path) -> None:
    """A bad (non-ASCII) metadata value raises before the file is written at all."""
    points, triangles, labels = _mesh_3d()
    filename = tmp_path / "mesh.rec"

    with pytest.raises(UnicodeEncodeError):
        iorec.save_rec(
            filename,
            points,
            triangles,
            labels,
            metadata={"spacing_unit": "µm"},
        )

    assert not filename.exists()


def test_metadata_round_trip_reuse(tmp_path: Path) -> None:
    """Feeding a loaded appendix dict straight back into save_rec doesn't corrupt it."""
    points, triangles, labels = _mesh_3d()
    filename_a = tmp_path / "a.rec"
    filename_b = tmp_path / "b.rec"

    iorec.save_rec(
        filename_a,
        points,
        triangles,
        labels,
        metadata={"coordinate_frame": "voxel_index"},
    )
    reused_metadata = iorec.load_rec_appendix(filename_a)
    assert reused_metadata is not None

    iorec.save_rec(filename_b, points, triangles, labels, metadata=reused_metadata)

    appendix_b = iorec.load_rec_appendix(filename_b)
    assert appendix_b is not None
    assert appendix_b["coordinate_frame"] == "voxel_index"
    with filename_b.open("rb") as f:
        content = f.read()
    assert content.count(b"written_by") == 1
    assert b"_version =" not in content


def test_load_rec_with_metadata(tmp_path: Path) -> None:
    """load_rec_with_metadata combines load_rec and load_rec_appendix."""
    points, triangles, labels = _mesh_3d()
    filename = tmp_path / "mesh.rec"
    metadata = {"coordinate_frame": "voxel_index", "spacing": (1.0, 0.325, 0.325)}

    iorec.save_rec(filename, points, triangles, labels, metadata=metadata)

    loaded_points, loaded_triangles, loaded_labels, loaded_metadata = (
        iorec.load_rec_with_metadata(filename)
    )
    assert np.array_equal(points, loaded_points)
    assert np.array_equal(triangles, loaded_triangles)
    assert np.array_equal(labels, loaded_labels)
    assert loaded_metadata is not None
    assert loaded_metadata["coordinate_frame"] == "voxel_index"
    assert loaded_metadata["spacing"] == (1.0, 0.325, 0.325)


def test_load_rec_with_metadata_none_when_absent(tmp_path: Path) -> None:
    """load_rec_with_metadata returns metadata=None for a file with no appendix."""
    points, triangles, labels = _mesh_3d()
    filename = tmp_path / "mesh.rec"

    iorec.save_rec(filename, points, triangles, labels)

    loaded_points, loaded_triangles, loaded_labels, loaded_metadata = (
        iorec.load_rec_with_metadata(filename)
    )
    assert np.array_equal(points, loaded_points)
    assert np.array_equal(triangles, loaded_triangles)
    assert np.array_equal(labels, loaded_labels)
    assert loaded_metadata is None


def test_load_2d_rec_with_metadata(tmp_path: Path) -> None:
    """load_2d_rec_with_metadata combines load_2d_rec and load_rec_appendix."""
    points, segments, labels = _mesh_2d()
    filename = tmp_path / "mesh2d.rec"
    metadata = {"coordinate_frame": "physical", "spacing": (0.5, 0.5)}

    iorec.save_2d_rec(filename, points, segments, labels, metadata=metadata)

    loaded_points, loaded_segments, loaded_labels, loaded_metadata = (
        iorec.load_2d_rec_with_metadata(filename)
    )
    assert np.array_equal(points, loaded_points)
    assert np.array_equal(segments, loaded_segments)
    assert np.array_equal(labels, loaded_labels)
    assert loaded_metadata is not None
    assert loaded_metadata["coordinate_frame"] == "physical"
    assert loaded_metadata["spacing"] == (0.5, 0.5)


def test_load_2d_rec_with_metadata_none_when_absent(tmp_path: Path) -> None:
    """load_2d_rec_with_metadata returns metadata=None for a file with no appendix."""
    points, segments, labels = _mesh_2d()
    filename = tmp_path / "mesh2d.rec"

    iorec.save_2d_rec(filename, points, segments, labels)

    loaded_points, loaded_segments, loaded_labels, loaded_metadata = (
        iorec.load_2d_rec_with_metadata(filename)
    )
    assert np.array_equal(points, loaded_points)
    assert np.array_equal(segments, loaded_segments)
    assert np.array_equal(labels, loaded_labels)
    assert loaded_metadata is None


@pytest.mark.parametrize("binary_mode", [False, True])
def test_load_rec_with_metadata_reads_file_once(
    tmp_path: Path,
    binary_mode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_rec_with_metadata reads the file exactly once.

    Unlike calling load_rec and load_rec_appendix separately, which would each read
    the whole file on their own.
    """
    points, triangles, labels = _mesh_3d()
    filename = tmp_path / "mesh.rec"
    iorec.save_rec(
        filename,
        points,
        triangles,
        labels,
        binary_mode=binary_mode,
        metadata={"coordinate_frame": "voxel_index"},
    )

    read_bytes_calls = 0
    original_read_bytes = Path.read_bytes

    def counting_read_bytes(self: Path) -> bytes:
        nonlocal read_bytes_calls
        read_bytes_calls += 1
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)

    iorec.load_rec_with_metadata(filename)

    # `read_bytes` itself opens the file once internally -- what matters is that the
    # file's content is fetched from disk exactly once, not twice (once streamed for
    # geometry, once via a separate `read_bytes()` for the appendix, as calling
    # `load_rec`+`load_rec_appendix` would do).
    assert read_bytes_calls == 1


@pytest.mark.parametrize("binary_mode", [False, True])
def test_load_2d_rec_with_metadata_reads_file_once(
    tmp_path: Path,
    binary_mode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_2d_rec_with_metadata reads the file exactly once.

    Unlike calling load_2d_rec and load_rec_appendix separately.
    """
    points, segments, labels = _mesh_2d()
    filename = tmp_path / "mesh2d.rec"
    iorec.save_2d_rec(
        filename,
        points,
        segments,
        labels,
        binary_mode=binary_mode,
        metadata={"coordinate_frame": "physical"},
    )

    read_bytes_calls = 0
    original_read_bytes = Path.read_bytes

    def counting_read_bytes(self: Path) -> bytes:
        nonlocal read_bytes_calls
        read_bytes_calls += 1
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)

    iorec.load_2d_rec_with_metadata(filename)

    # See test_load_rec_with_metadata_reads_file_once for why only read_bytes matters.
    assert read_bytes_calls == 1


def test_load_rec_appendix_missing_file_raises() -> None:
    """load_rec_appendix on a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        iorec.load_rec_appendix("tests/assets/NOT_A_FILE.rec")

    assert (
        exc_info.value.args[0]
        == f"{Path('tests/assets/NOT_A_FILE.rec').resolve()} not found."
    )
