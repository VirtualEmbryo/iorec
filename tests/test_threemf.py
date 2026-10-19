"""3MF files: closed outward cell surfaces, printing metadata, round trips."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable
from xml.etree import ElementTree

import numpy as np
import pytest
from conftest import Mesh

import iorec
from iorec._topology import _open_half_edges, _signed_volumes

CORE = "{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}"


def _model(path: Path) -> ElementTree.Element:
    with zipfile.ZipFile(path) as archive:
        return ElementTree.fromstring(archive.read("3D/3dmodel.model"))  # noqa: S314


def test_round_trip_gives_the_same_mesh(
    tmp_path: Path,
    cells: Mesh,
    same_mesh: Callable[[Mesh, Mesh], bool],
) -> None:
    """Merging the cell surfaces back gives the same labelled triangles."""
    iorec.save_3mf(tmp_path / "c.3mf", *cells, scale=3.0, unit="micron")
    loaded = iorec.load_3mf(tmp_path / "c.3mf", validate=True)
    assert same_mesh(loaded, cells)
    assert sorted(map(tuple, loaded[0].tolist())) == sorted(
        map(tuple, cells[0].tolist()),
    )


def test_cells_are_closed_and_outward(tmp_path: Path, cells: Mesh) -> None:
    """Each object is a closed surface whose enclosed volume is the cell volume."""
    iorec.save_3mf(tmp_path / "c.3mf", *cells)
    objects = [
        o
        for o in _model(tmp_path / "c.3mf").iter(f"{CORE}object")
        if o.find(f"{CORE}mesh") is not None
    ]
    volumes = _signed_volumes(*cells)
    assert len(objects) == 16
    for obj in objects:
        vertices = np.array(
            [[float(v.get(a)) for a in "xyz"] for v in obj.iter(f"{CORE}vertex")],
        )
        triangles = np.array(
            [
                [int(t.get(f"v{k}")) for k in (1, 2, 3)]
                for t in obj.iter(f"{CORE}triangle")
            ],
        )
        label = int(obj.get("name").split()[1])
        assert _open_half_edges(triangles) == 0
        enclosed = _signed_volumes(
            vertices,
            triangles,
            np.tile([label, 0], (len(triangles), 1)),
        )
        assert enclosed[label] == pytest.approx(volumes[label], rel=1e-12)


def test_printing_metadata(tmp_path: Path, cells: Mesh) -> None:
    """Unit, scale and placement in the build item; Cura extruders; colours."""
    points = cells[0]
    iorec.save_3mf(
        tmp_path / "p.3mf",
        *cells,
        unit="micron",
        scale=500.0,
        extruders={3: 1, 4: 0},
    )
    model = _model(tmp_path / "p.3mf")
    assert model.get("unit") == "micron"
    item = model.find(f"{CORE}build/{CORE}item")
    transform = np.array(item.get("transform").split(), dtype=float).reshape(4, 3)
    assert np.array_equal(transform[:3], 500.0 * np.eye(3))
    placed = points @ transform[:3] + transform[3]
    assert placed.min(axis=0) == pytest.approx([0, 0, 0], abs=1e-9)
    names = {}
    for obj in model.iter(f"{CORE}object"):
        for meta in obj.iter(f"{CORE}metadata"):
            names.setdefault(obj.get("name"), {})[meta.get("name")] = meta.text
    assert names["cell 3"] == {"iorec:label": "3", "cura:extruder_nr": "1"}
    assert names["cell 4"]["cura:extruder_nr"] == "0"
    assert "cura:extruder_nr" not in names["cell 5"]
    assert len(model.find(f"{CORE}resources/{CORE}basematerials")) == 16


def test_open_surfaces_warn(
    tmp_path: Path,
    open_cells: Mesh,
    same_mesh: Callable[[Mesh, Mesh], bool],
) -> None:
    """A cell surface that is not closed is written, with a warning per cell."""
    with pytest.warns(UserWarning, match="not closed") as record:
        iorec.save_3mf(tmp_path / "o.3mf", *open_cells)
    assert len(record) == 3
    assert same_mesh(iorec.load_3mf(tmp_path / "o.3mf"), open_cells)


def test_foreign_file_with_transforms(tmp_path: Path) -> None:
    """Objects of other writers are labels 1, 2, ...; their transforms are applied."""
    tetra = (
        '<mesh><vertices><vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/>'
        '<vertex x="0" y="1" z="0"/><vertex x="0" y="0" z="1"/></vertices><triangles>'
        '<triangle v1="0" v2="2" v3="1"/><triangle v1="0" v2="1" v3="3"/>'
        '<triangle v1="1" v2="2" v3="3"/><triangle v1="2" v2="0" v3="3"/>'
        "</triangles></mesh>"
    )
    model = (
        f'<?xml version="1.0"?><model unit="millimeter" xmlns="{CORE[1:-1]}">'
        f'<resources><object id="1" type="model">{tetra}</object></resources><build>'
        '<item objectid="1"/><item objectid="1" transform="2 0 0 0 2 0 0 0 2 5 0 0"/>'
        "</build></model>"
    )
    with zipfile.ZipFile(tmp_path / "f.3mf", "w") as archive:
        archive.writestr("3D/3dmodel.model", model)
    points, triangles, labels = iorec.load_3mf(tmp_path / "f.3mf")
    assert len(points) == 8
    assert points.max(axis=0).tolist() == [7.0, 2.0, 2.0]
    volumes = _signed_volumes(points, triangles, labels)
    assert volumes[1] == pytest.approx(1 / 6)
    assert volumes[2] == pytest.approx(8 / 6)


def test_bad_options(tmp_path: Path, cells: Mesh) -> None:
    """Unknown units and non-positive scales are refused before writing."""
    with pytest.raises(ValueError, match="unit must be one of"):
        iorec.save_3mf(tmp_path / "x.3mf", *cells, unit="mm")
    with pytest.raises(ValueError, match="scale must be a positive number"):
        iorec.save_3mf(tmp_path / "x.3mf", *cells, scale=0.0)
    assert not (tmp_path / "x.3mf").exists()


def test_not_a_3mf_package(tmp_path: Path) -> None:
    """A file that is not a zip package raises MeshFormatError."""
    (tmp_path / "x.3mf").write_text("not a zip")
    with pytest.raises(iorec.MeshFormatError, match="not a readable 3MF package"):
        iorec.load_3mf(tmp_path / "x.3mf")


def _foreign_model(tmp_path: Path, objects: str, build: str) -> Path:
    model = (
        f'<?xml version="1.0"?><model unit="millimeter" xmlns="{CORE[1:-1]}">'
        f"<resources>{objects}</resources><build>{build}</build></model>"
    )
    path = tmp_path / "g.3mf"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("3D/3dmodel.model", model)
    return path


TETRA = (
    '<object id="1"><mesh><vertices><vertex x="0" y="0" z="0"/>'
    '<vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/>'
    '<vertex x="0" y="0" z="1"/></vertices><triangles>'
    '<triangle v1="0" v2="2" v3="1"/><triangle v1="0" v2="1" v3="3"/>'
    '<triangle v1="1" v2="2" v3="3"/><triangle v1="2" v2="0" v3="3"/>'
    "</triangles></mesh></object>"
)


def test_components_compose_their_transforms(tmp_path: Path) -> None:
    """A component is placed by its own transform, then by that of the build item."""
    nested = (
        TETRA + '<object id="2"><components><component objectid="1" '
        'transform="1 0 0 0 1 0 0 0 1 0 3 0"/></components></object>'
    )
    path = _foreign_model(
        tmp_path,
        nested,
        '<item objectid="2" transform="2 0 0 0 2 0 0 0 2 5 0 0"/>',
    )
    points, _, _ = iorec.load_3mf(path)
    assert points.min(axis=0).tolist() == [5.0, 6.0, 0.0]
    assert points.max(axis=0).tolist() == [7.0, 8.0, 2.0]


def test_bad_object_references(tmp_path: Path) -> None:
    """A missing object and an object containing itself are reported separately."""
    path = _foreign_model(tmp_path, TETRA, '<item objectid="7"/>')
    with pytest.raises(iorec.MeshFormatError, match="missing object '7'"):
        iorec.load_3mf(path)
    loop = (
        '<object id="2"><components><component objectid="3"/></components></object>'
        '<object id="3"><components><component objectid="2"/></components></object>'
    )
    path = _foreign_model(tmp_path, TETRA + loop, '<item objectid="2"/>')
    with pytest.raises(iorec.MeshFormatError, match="contains itself"):
        iorec.load_3mf(path)
