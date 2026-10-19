"""Format chosen by suffix, conversions through every format, command line."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from conftest import ASSETS, Mesh

import iorec
from iorec.__main__ import main


def test_conversion_chain_keeps_the_mesh(
    tmp_path: Path,
    cells: Mesh,
    same_mesh: Callable[[Mesh, Mesh], bool],
) -> None:
    """.rec -> .fe -> .ply -> .obj -> .3mf -> .rec gives the same labelled triangles."""
    chain = ["a.rec", "b.fe", "c.ply", "d.obj", "e.3mf", "f.rec"]
    iorec.save_rec(tmp_path / chain[0], *cells, binary_mode=True)
    for source, target in zip(chain, chain[1:]):
        iorec.convert(tmp_path / source, tmp_path / target)
    assert same_mesh(iorec.load_mesh(tmp_path / chain[-1]), cells)


def test_unknown_suffix(tmp_path: Path, cells: Mesh) -> None:
    """A suffix without a reader or writer is refused with the list of suffixes."""
    with pytest.raises(ValueError, match=r"\.3mf, \.arec, \.fe, \.obj, \.ply, \.rec"):
        iorec.save_mesh(tmp_path / "x.stl", *cells)
    with pytest.raises(ValueError, match="cannot read"):
        iorec.load_mesh(tmp_path / "x.vtk")


def test_rec_metadata_is_kept_or_reported(tmp_path: Path, cells: Mesh) -> None:
    """The appendix follows a .rec -> .rec conversion; other targets warn."""
    metadata = {"spacing": (1.0, 0.5, 0.5), "axis_order": "zyx"}
    iorec.save_rec(tmp_path / "m.rec", *cells, metadata=metadata)
    iorec.convert(
        tmp_path / "m.rec",
        tmp_path / "n.arec",
        save_options={"binary_mode": True},
    )
    copied = iorec.load_rec_appendix(tmp_path / "n.arec")
    assert copied["spacing"] == (1.0, 0.5, 0.5)
    assert copied["axis_order"] == "zyx"
    with pytest.warns(UserWarning, match="metadata appendix"):
        iorec.convert(tmp_path / "m.rec", tmp_path / "m.ply")


def test_command_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """mt2se-like use: tensions from a file, volumes held; then a 3MF with extruders."""
    source = ASSETS / "binary.rec"
    fe, tension_file = tmp_path / "c.fe", tmp_path / "tensions.txt"
    convert_to_fe = [str(source), str(fe), "--fix-volumes", "--tension-file"]
    tension_file.write_text("".join(f"0 {k} 0.5\n" for k in range(1, 17)))
    assert main([*convert_to_fe, str(tension_file)]) == 1
    assert "no tension given for the interfaces" in capsys.readouterr().err
    pairs = sorted({tuple(sorted(p)) for p in iorec.load_rec(source)[2].tolist()})
    rows = [f"{a} {b} {0.5 if a == 0 else 1.0}\n" for a, b in pairs]
    tension_file.write_text("".join(rows))
    assert main([*convert_to_fe, str(tension_file)]) == 0
    _, _, labels, attributes = iorec.load_fe_with_attributes(fe)
    expected = np.where((labels == 0).any(axis=1), 0.5, 1.0)
    assert np.array_equal(attributes["tension"], expected)
    assert "set body[16] target body[16].volume" in fe.read_text()
    target = tmp_path / "c.3mf"
    with pytest.warns(UserWarning, match="facet tensions and body volumes"):
        status = main([str(fe), str(target), "--scale", "4", "--extruder", "2=1"])
    assert status == 0
    with zipfile.ZipFile(target) as archive:
        model = archive.read("3D/3dmodel.model").decode()
    assert '<metadata name="cura:extruder_nr" preserve="1">1</metadata>' in model


@pytest.mark.parametrize("suffix", [".fe", ".ply", ".obj", ".3mf"])
def test_writers_refuse_invalid_mesh(tmp_path: Path, suffix: str) -> None:
    """Every writer applies the checks of the readers, raising a plain ValueError."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    bad = {
        "outside": (points, [[0, 1, 3]], [[1, 0]]),
        "negative label": (points, [[0, 1, 2]], [[-1, 0]]),
        "itself": (points, [[0, 1, 2]], [[1, 1]]),
        "non-finite": (points * np.nan, [[0, 1, 2]], [[1, 0]]),
    }
    for match, mesh in bad.items():
        with pytest.raises(ValueError, match=match):
            iorec.save_mesh(tmp_path / f"bad{suffix}", *mesh)


def test_exterior_label_follows_the_conversion(tmp_path: Path, cells: Mesh) -> None:
    """A stored exterior label other than 0 reaches the writer of the target."""
    points, triangles, labels = cells
    shifted = labels + 1
    iorec.save_3mf(tmp_path / "a.3mf", points, triangles, shifted, exterior_label=1)
    iorec.convert(tmp_path / "a.3mf", tmp_path / "b.obj")
    assert "iorec exterior_label 1" in (tmp_path / "b.obj").read_text()
    assert iorec.load_obj(tmp_path / "b.obj")[2].min() == 1
    iorec.convert(
        tmp_path / "b.obj",
        tmp_path / "c.3mf",
        save_options={"exterior_label": 1},
    )
    assert iorec.load_3mf(tmp_path / "c.3mf")[2].min() == 1
    iorec.convert(tmp_path / "a.3mf", tmp_path / "d.fe")
    assert "label 1 is the exterior" in (tmp_path / "d.fe").read_text()


def test_evolver_extras_are_kept_or_reported(tmp_path: Path, cells: Mesh) -> None:
    """Tensions and volume targets follow .fe -> .fe; other targets warn."""
    iorec.save_fe(tmp_path / "a.fe", *cells, tensions=0.5, volumes={3: 7.5})
    iorec.convert(tmp_path / "a.fe", tmp_path / "b.fe")
    _, _, _, attributes = iorec.load_fe_with_attributes(tmp_path / "b.fe")
    assert np.all(attributes["tension"] == 0.5)
    assert attributes["volume"] == {3: 7.5}
    with pytest.warns(UserWarning, match="facet tensions and body volumes"):
        iorec.convert(tmp_path / "a.fe", tmp_path / "a.ply")
    iorec.save_fe(tmp_path / "plain.fe", *cells)
    iorec.convert(tmp_path / "plain.fe", tmp_path / "plain.ply")
