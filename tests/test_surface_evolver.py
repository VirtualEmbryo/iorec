"""Surface Evolver datafiles and dumps: orientation, labels, round trips, errors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import ASSETS, Mesh, signed_volumes

import iorec

# A triangle (three vertices, three edges), to which each case adds facets.
TRIANGLE = "vertices\n1 0 0 0\n2 1 0 0\n3 0 1 0\nedges\n1 1 2\n2 2 3\n3 3 1\n"


def _area(points: np.ndarray, triangles: np.ndarray) -> float:
    p0, p1, p2 = (points[triangles[:, k]] for k in range(3))
    return float(np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum() / 2)


def test_round_trip_is_exact(tmp_path: Path, cells: Mesh) -> None:
    """Writing then reading a datafile gives back the same arrays, bit for bit."""
    iorec.save_fe(tmp_path / "cells.fe", *cells)
    points, triangles, labels = iorec.load_fe(tmp_path / "cells.fe", validate=True)
    assert np.array_equal(points, cells[0])
    assert np.array_equal(triangles, cells[1])
    assert np.array_equal(labels, cells[2])


def test_open_surfaces_round_trip(tmp_path: Path, open_cells: Mesh) -> None:
    """Facets with the exterior on their positive side keep their orientation."""
    iorec.save_fe(tmp_path / "open.fe", *open_cells)
    loaded = iorec.load_fe(tmp_path / "open.fe")
    for read, written in zip(loaded, open_cells):
        assert np.array_equal(read, written)


def test_bodies_follow_outward_normals(tmp_path: Path, cells: Mesh) -> None:
    """Body k lists +f when facet f points out of label k, -f when it points in."""
    points, triangles, labels = cells
    iorec.save_fe(tmp_path / "cells.fe", points, triangles, labels)
    text = (tmp_path / "cells.fe").read_text()
    body_3 = text.split("\nbodies\n")[1].split("\n3 ")[1].split("\n4 ")[0]
    signed = [int(t) for t in body_3.replace("\\", " ").split()]
    expected = [i + 1 for i in np.flatnonzero(labels[:, 0] == 3)]
    expected += [-(i + 1) for i in np.flatnonzero(labels[:, 1] == 3)]
    assert signed == expected


def test_quadrilaterals_split_around_centre() -> None:
    """Two unit cubes given by squares: 4 triangles per square, volumes 1, area 11."""
    points, triangles, labels, attributes = iorec.load_fe_with_attributes(
        ASSETS / "two_cubes.fe",
        validate=True,
    )
    assert triangles.shape == (44, 3)
    assert points.shape == (12 + 11, 3)
    assert points[12 + 5].tolist() == [1.0, 0.5, 0.5]  # centre of the shared square
    assert (attributes["vertex_id"][12:] == 0).all()
    assert _area(points, triangles) == pytest.approx(11.0, rel=1e-15)
    volumes = signed_volumes(points, triangles, labels)
    assert volumes[1] == pytest.approx(1.0, rel=1e-14)
    assert volumes[2] == pytest.approx(1.0, rel=1e-14)
    assert attributes["volume"] == {1: 1.0, 2: 1.0}
    shared = (labels == [1, 2]).all(axis=1)
    assert shared.sum() == 4
    assert np.allclose(points[triangles[shared]][..., 0], 1.0)


def test_evolver_dump_is_read() -> None:
    """A dump of Surface Evolver 2.70 after refining and 12 iterations of `g`.

    Evolver printed total_area 9.26947364644248 and both body volumes
    1.00000000083127 before dumping (15 significant digits).
    """
    points, triangles, labels, attributes = iorec.load_fe_with_attributes(
        ASSETS / "two_cubes_evolved.dmp",
        validate=True,
    )
    assert triangles.shape == (176, 3)
    assert _area(points, triangles) == pytest.approx(9.26947364644248, rel=1e-13)
    volumes = signed_volumes(points, triangles, labels)
    assert volumes[1] == pytest.approx(1.00000000083127, rel=1e-13)
    assert volumes[2] == pytest.approx(1.00000000083127, rel=1e-13)
    assert attributes["volume"] == {1: 1.0, 2: 1.0}
    assert sorted(map(tuple, np.unique(labels, axis=0))) == [(1, 0), (1, 2), (2, 0)]


def test_tensions_by_interface(tmp_path: Path, cells: Mesh) -> None:
    """A {(a, b): tension} mapping gives each facet the tension of its label pair."""
    labels = cells[2]
    pairs = {tuple(sorted(pair)) for pair in labels.tolist()}
    tensions = {pair: (0.5 if 0 in pair else 1.0 + pair[0] / 100) for pair in pairs}
    iorec.save_fe(tmp_path / "t.fe", *cells, tensions=tensions)
    _, _, read_labels, attributes = iorec.load_fe_with_attributes(tmp_path / "t.fe")
    expected = [tensions[tuple(sorted(pair))] for pair in read_labels.tolist()]
    assert np.array_equal(attributes["tension"], expected)


def test_missing_interface_tension_is_refused(tmp_path: Path, cells: Mesh) -> None:
    """Evolver would silently use tension 1 for a pair missing from the mapping."""
    with pytest.raises(ValueError, match="no tension given"):
        iorec.save_fe(tmp_path / "t.fe", *cells, tensions={(0, 1): 2.0})


def test_volume_targets(tmp_path: Path, cells: Mesh) -> None:
    """Given targets are body attributes; "current" uses Evolver's own volume."""
    iorec.save_fe(tmp_path / "v.fe", *cells, volumes={3: 7.5, 4: 7.0})
    _, _, _, attributes = iorec.load_fe_with_attributes(tmp_path / "v.fe")
    assert attributes["volume"] == {3: 7.5, 4: 7.0}
    with pytest.raises(ValueError, match=r"labels \[0, 17\], which are not bodies"):
        iorec.save_fe(tmp_path / "v.fe", *cells, volumes={0: 1.0, 17: 1.0})
    iorec.save_fe(tmp_path / "c.fe", *cells, volumes="current", read_commands="g 5")
    tail = (tmp_path / "c.fe").read_text().split("\nread\n")[1].splitlines()
    assert tail[0] == "set body[1] target body[1].volume"
    assert len(tail) == 17
    assert tail[-1] == "g 5"


def test_sparse_labels_are_body_numbers(tmp_path: Path, cells: Mesh) -> None:
    """Labels need not be consecutive: body k is label k (keep_originals)."""
    points, triangles, labels = cells
    sparse = np.where(labels > 0, 10 * labels + 3, 0)
    iorec.save_fe(tmp_path / "s.fe", points, triangles, sparse)
    assert np.array_equal(iorec.load_fe(tmp_path / "s.fe")[2], sparse)


def test_exterior_label(tmp_path: Path, cells: Mesh) -> None:
    """Another label can be the exterior; label 0 then cannot be a body."""
    points, triangles, labels = cells
    with pytest.raises(ValueError, match="label 0 cannot be a Surface Evolver body"):
        iorec.save_fe(tmp_path / "e.fe", points, triangles, labels, exterior_label=5)
    shifted = labels + 1
    iorec.save_fe(tmp_path / "e.fe", points, triangles, shifted, exterior_label=1)
    read = iorec.load_fe(tmp_path / "e.fe", exterior_label=1)[2]
    assert np.array_equal(read, shifted)


def test_lexical_rules(tmp_path: Path) -> None:
    """Comments, line splices, case, commas and Fortran exponents as in Evolver."""
    text = """// header comment
    SOAPFILM  /* a block
    comment over two lines */
    VERTICES
    1  0, 0, 0
    2  1.0D0 0 0 fixed   // trailing comment
    3  0 1 0
    4  0 0 +1E0
    EDGES
    1 1 2;
    2 2 3
    3 3 1
    4 1 4
    5 2 4
    6 3 4
    FACETS
    1 -3 -2 -1 tension 2.5
    2 1 5 -4
    3 2 6 -5 color red
    4 3 4 -6
    BODIES
    1 1 2 3 \\
      4 volume 0.5
    read
    1 0 0 0 this is not read
    """
    (tmp_path / "tetra.fe").write_text(text)
    points, triangles, labels, attributes = iorec.load_fe_with_attributes(
        tmp_path / "tetra.fe",
        validate=True,
    )
    assert points.tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert triangles.tolist() == [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
    assert (labels == [1, 0]).all()
    assert attributes["tension"].tolist() == [2.5, 1.0, 1.0, 1.0]
    assert signed_volumes(points, triangles, labels)[1] == pytest.approx(1 / 6)


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("STRING", "string model"),
        ("QUADRATIC", "quadratic model"),
        ("LAGRANGE 2", "Lagrange model"),
        ("TORUS_FILLED", "torus model"),
        ("SPACE_DIMENSION 4", "space_dimension 4"),
    ],
)
def test_unsupported_models(tmp_path: Path, header: str, message: str) -> None:
    """Models other than the linear 3D soapfilm are refused, not misread."""
    (tmp_path / "m.fe").write_text(f"{header}\nvertices\n1 0 0 0\n")
    with pytest.raises(iorec.MeshFormatError, match=message):
        iorec.load_fe(tmp_path / "m.fe")


def test_header_procedures_are_skipped(tmp_path: Path) -> None:
    """A keyword at the start of a line inside a header procedure is not a model."""
    text = "gogo := {\nlagrange 2\n}\nvertices\n1 0 0 0\n"
    (tmp_path / "p.fe").write_text(text)
    assert iorec.load_fe(tmp_path / "p.fe")[0].shape == (1, 3)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("vertices\n1 sqrt(2) 0 0\n", "three numeric coordinates"),
        ("vertices\n1 0.5 boundary 1\n", "parametric boundary"),
        (
            "vertices\n1 0 0 0\n2 1 0 0\nedges\n1 1 2\nfaces\n1 1 -1\n",
            "fewer than three",
        ),
        (
            TRIANGLE + "faces\n1 1 2 -3\n",
            "edges that do not close",
        ),
        (
            TRIANGLE + "faces\n1 1 2 9\n",
            "undefined edge",
        ),
        ("vertices\n1 0 0 0\nedges\n1 1 2\n", "undefined vertex"),
        ("vertices\n1 0 0 0\n1 1 0 0\n", "vertex 1 is defined twice"),
        ("vertices\n1 0 0 0\nbodies\n1 7\n", "undefined facet"),
    ],
)
def test_malformed_datafiles(tmp_path: Path, body: str, message: str) -> None:
    """Malformed element lists raise MeshFormatError naming the problem."""
    (tmp_path / "bad.fe").write_text(body)
    with pytest.raises(iorec.MeshFormatError, match=message):
        iorec.load_fe(tmp_path / "bad.fe")


def test_facet_on_one_side_of_two_bodies(tmp_path: Path) -> None:
    """A facet with positive sign in two bodies is inconsistent."""
    text = (
        "vertices\n1 0 0 0\n2 1 0 0\n3 0 1 0\nedges\n1 1 2\n2 2 3\n3 3 1\n"
        "faces\n1 1 2 3\nbodies\n1 1\n2 1\n"
    )
    (tmp_path / "two.fe").write_text(text)
    with pytest.raises(iorec.MeshFormatError, match="two bodies on one side"):
        iorec.load_fe(tmp_path / "two.fe")


def test_facets_without_body_fail_validation(tmp_path: Path) -> None:
    """A soap film bounding no body gets the exterior label on both sides."""
    text = (
        "vertices\n1 0 0 0\n2 1 0 0\n3 0 1 0\n"
        "edges\n1 1 2\n2 2 3\n3 3 1\nfaces\n1 1 2 3\n"
    )
    (tmp_path / "film.fe").write_text(text)
    assert iorec.load_fe(tmp_path / "film.fe")[2].tolist() == [[0, 0]]
    with pytest.raises(iorec.MeshFormatError, match="separates label 0 from itself"):
        iorec.load_fe(tmp_path / "film.fe", validate=True)
