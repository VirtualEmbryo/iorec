"""Check written datafiles with Surface Evolver itself, when it is installed.

The Evolver executable is taken from the IOREC_EVOLVER environment variable, or found
on the PATH as `evolver` or `evolver-nox-d`; without it these tests are skipped.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import Mesh

import iorec
from iorec._topology import _signed_volumes

EVOLVER = os.environ.get("IOREC_EVOLVER") or next(
    (shutil.which(name) for name in ("evolver", "evolver-nox-d") if shutil.which(name)),
    None,
)
pytestmark = pytest.mark.skipif(EVOLVER is None, reason="Surface Evolver not found")


def _run_evolver(datafile: Path, commands: str) -> str:
    assert EVOLVER is not None
    result = subprocess.run(  # noqa: S603
        [EVOLVER, datafile.name],
        input=commands + "\nq\nq\n",
        capture_output=True,
        text=True,
        timeout=120,
        cwd=datafile.parent,
        check=False,
    )
    return result.stdout


def test_evolver_measures_the_same_volumes_and_area(
    tmp_path: Path,
    cells: Mesh,
) -> None:
    """Evolver's body volumes and total area equal iorec's, to 1e-13 relative."""
    points, triangles, labels = cells
    iorec.save_fe(tmp_path / "cells.fe", points, triangles, labels)
    output = _run_evolver(
        tmp_path / "cells.fe",
        'printf "AREA %.17g\\n", total_area\n'
        'foreach body bb do printf "BODY %d %.17g\\n", bb.id, bb.volume\n'
        "C\n",
    )
    rows = [line.split() for line in output.splitlines()]
    area = next(float(r[1]) for r in rows if r[:1] == ["AREA"])
    bodies = {int(r[1]): float(r[2]) for r in rows if r[:1] == ["BODY"]}
    p0, p1, p2 = (points[triangles[:, k]] for k in range(3))
    expected_area = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum() / 2
    assert area == pytest.approx(expected_area, rel=1e-13)
    volumes = _signed_volumes(points, triangles, labels)
    assert sorted(bodies) == list(range(1, 17))
    for label, volume in bodies.items():
        assert volume == pytest.approx(volumes[label], rel=1e-13)
    assert "Checks completed." in output
    assert "ERROR" not in output


def test_evolved_dump_reads_back(tmp_path: Path, cells: Mesh) -> None:
    """After `g 5` at held volumes, the dump has the same topology and volumes."""
    iorec.save_fe(tmp_path / "cells.fe", *cells, volumes="current")
    _run_evolver(tmp_path / "cells.fe", 'g 5\ndump "cells.dmp"\n')
    points, triangles, labels, attributes = iorec.load_fe_with_attributes(
        tmp_path / "cells.dmp",
        validate=True,
    )
    assert np.array_equal(triangles, cells[1])
    assert np.array_equal(labels, cells[2])
    assert not np.array_equal(points, cells[0])
    volumes = _signed_volumes(points, triangles, labels)
    for label, target in attributes["volume"].items():
        assert volumes[label] == pytest.approx(target, rel=1e-6)
