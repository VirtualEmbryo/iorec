"""OBJ files: one object per interface or per cell, colours, other writers' files."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from conftest import Mesh, signed_volumes

import iorec
from iorec._topology import _cell_shells


@pytest.mark.parametrize("group_by", ["interface", "cell"])
def test_round_trip_gives_the_same_mesh(
    tmp_path: Path,
    cells: Mesh,
    same_mesh: Callable[[Mesh, Mesh], bool],
    group_by: str,
) -> None:
    """Same points, and the same labelled triangles up to order and orientation."""
    iorec.save_obj(tmp_path / "c.obj", *cells, group_by=group_by)
    loaded = iorec.load_obj(tmp_path / "c.obj", validate=True)
    assert np.array_equal(loaded[0], cells[0])
    assert same_mesh(loaded, cells)


def test_open_surfaces_by_cell(
    tmp_path: Path,
    open_cells: Mesh,
    same_mesh: Callable[[Mesh, Mesh], bool],
) -> None:
    """Cell surfaces need not be closed to be merged back."""
    iorec.save_obj(tmp_path / "o.obj", *open_cells, group_by="cell")
    assert same_mesh(iorec.load_obj(tmp_path / "o.obj"), open_cells)


def test_objects_and_materials(tmp_path: Path, cells: Mesh) -> None:
    """One object and one material per interface; cell surfaces are oriented outward."""
    iorec.save_obj(tmp_path / "i.obj", *cells)
    text = (tmp_path / "i.obj").read_text()
    n_pairs = len(np.unique(np.sort(cells[2], axis=1), axis=0))
    assert text.count("\no interface_") == n_pairs
    assert "mtllib i.mtl\n" in text
    mtl = (tmp_path / "i.mtl").read_text()
    assert mtl.count("newmtl interface_") == n_pairs
    iorec.save_obj(tmp_path / "c.obj", *cells, group_by="cell", write_mtl=False)
    assert not (tmp_path / "c.mtl").exists()
    points, _, _ = cells
    cell_3 = (tmp_path / "c.obj").read_text().split("o cell_3\n")[1].split("\no ")[0]
    faces = [[int(v) - 1 for v in line.split()[1:]] for line in cell_3.splitlines()[1:]]
    volume = signed_volumes(points, np.array(faces), np.tile([3, 0], (len(faces), 1)))
    assert volume[3] == pytest.approx(signed_volumes(*cells)[3], rel=1e-12)
    assert volume[3] > 0


def test_foreign_file_with_quads_and_negative_indices(tmp_path: Path) -> None:
    """Objects without iorec names are cells 1, 2, ...; v/vt/vn and quads are read."""
    text = (
        "# exported elsewhere\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\n"
        "o first\nf 1/1/1 2/1/1 3/1/1 4/1/1\no second\nf -2//1 -3//1 -4//1\n"
    )
    (tmp_path / "f.obj").write_text(text)
    points, triangles, labels = iorec.load_obj(tmp_path / "f.obj")
    assert points.shape == (4, 3)
    assert triangles.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert labels.tolist() == [[1, 2], [1, 0]]


def test_blender_style_duplicated_vertices(
    tmp_path: Path,
    cells: Mesh,
    same_mesh: Callable[[Mesh, Mesh], bool],
) -> None:
    """Each cell object with its own copy of the vertices: merge_vertices rebuilds."""
    points, triangles, labels = cells
    lines, offset = [], 0
    for label, shell in _cell_shells(triangles, labels, 0):
        used, local = np.unique(shell, return_inverse=True)
        lines += [f"o cell_{label}.001"]
        lines += [f"v {x!r} {y!r} {z!r}" for x, y, z in points[used].tolist()]
        lines += [f"f {a} {b} {c}" for a, b, c in local.reshape(-1, 3) + 1 + offset]
        offset += len(used)
    (tmp_path / "b.obj").write_text("\n".join(lines) + "\n")
    merged = iorec.load_obj(tmp_path / "b.obj", merge_vertices=True, validate=True)
    assert len(merged[0]) == len(points)
    assert same_mesh(merged, cells)
    unmerged = iorec.load_obj(tmp_path / "b.obj")
    assert len(unmerged[1]) > len(triangles)


def test_bad_grouping(tmp_path: Path, cells: Mesh) -> None:
    """Only the two groupings are accepted."""
    with pytest.raises(ValueError, match="group_by"):
        iorec.save_obj(tmp_path / "x.obj", *cells, group_by="material")
