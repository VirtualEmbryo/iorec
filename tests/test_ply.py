"""PLY files: exact round trips in both encodings, other writers' files, errors."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from conftest import Mesh

import iorec


@pytest.mark.parametrize("binary_mode", [True, False])
@pytest.mark.parametrize("colors", [False, True])
def test_round_trip_is_exact(
    tmp_path: Path,
    cells: Mesh,
    binary_mode: bool,
    colors: bool,
) -> None:
    """The same arrays come back bit for bit: binary or text, colours or not."""
    iorec.save_ply(tmp_path / "c.ply", *cells, binary_mode=binary_mode, colors=colors)
    points, triangles, labels = iorec.load_ply(tmp_path / "c.ply", validate=True)
    assert np.array_equal(points, cells[0])
    assert np.array_equal(triangles, cells[1])
    assert np.array_equal(labels, cells[2])
    assert (points.dtype, triangles.dtype, labels.dtype) == (
        np.float64,
        np.int64,
        np.int64,
    )


def test_header_declares_labels(tmp_path: Path, open_cells: Mesh) -> None:
    """The labels are face properties named label0 and label1."""
    iorec.save_ply(tmp_path / "o.ply", *open_cells, binary_mode=False)
    header = (tmp_path / "o.ply").read_text().split("end_header")[0]
    assert "element face 318\nproperty list uchar int vertex_indices\n" in header
    assert "property int label0\nproperty int label1\n" in header


def test_big_endian_file_with_quads_and_extra_element(tmp_path: Path) -> None:
    """Another writer's file: big-endian, float vertices, a quad, an extra element."""
    header = (
        "ply\nformat binary_big_endian 1.0\ncomment made by hand\n"
        "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element face 1\nproperty list uchar uint vertex_index\n"
        "element edge 1\nproperty int vertex1\nproperty int vertex2\nend_header\n"
    )
    body = b"".join(
        struct.pack(">fffBBB", *xyz, 255, 0, 0)
        for xyz in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    )
    body += struct.pack(">BIIII", 4, 0, 1, 2, 3) + struct.pack(">ii", 0, 1)
    (tmp_path / "q.ply").write_bytes(header.encode() + body)
    points, triangles, labels = iorec.load_ply(tmp_path / "q.ply")
    assert points.tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    assert triangles.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert labels.tolist() == [[1, 0], [1, 0]]


def test_text_polygons_of_different_sizes(tmp_path: Path) -> None:
    """Faces of 3 and 4 vertices in a text file; quads split around a vertex."""
    text = (
        "ply\nformat ascii 1.0\nelement vertex 5\nproperty double x\n"
        "property double y\nproperty double z\nelement face 2\n"
        "property list uchar int vertex_indices\nproperty int label0\n"
        "property int label1\nend_header\n"
        "0 0 0\n1 0 0\n1 1 0\n0 1 0\n0.5 0.5 1\n4 0 1 2 3 2 0\n3 0 1 4 2 1\n"
    )
    (tmp_path / "p.ply").write_text(text)
    _, triangles, labels = iorec.load_ply(tmp_path / "p.ply")
    assert triangles.tolist() == [[0, 1, 2], [0, 2, 3], [0, 1, 4]]
    assert labels.tolist() == [[2, 0], [2, 0], [2, 1]]


def test_truncated_binary_file(tmp_path: Path, cells: Mesh) -> None:
    """A binary file cut in its face block raises MeshFormatError."""
    iorec.save_ply(tmp_path / "c.ply", *cells)
    data = (tmp_path / "c.ply").read_bytes()
    (tmp_path / "cut.ply").write_bytes(data[:-100])
    with pytest.raises(iorec.MeshFormatError, match="truncated or malformed"):
        iorec.load_ply(tmp_path / "cut.ply")


def test_not_a_ply_file(tmp_path: Path) -> None:
    """A file without a PLY header is refused."""
    (tmp_path / "x.ply").write_text("solid nothing\n")
    with pytest.raises(iorec.MeshFormatError, match="not a PLY file"):
        iorec.load_ply(tmp_path / "x.ply")
