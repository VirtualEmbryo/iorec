"""Every writer to every reader: save in format A, convert to B, load, compare."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
from conftest import ASSETS, Mesh

import iorec

# name -> (suffix, writer options)
VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "rec-text": (".rec", {"binary_mode": False}),
    "rec-binary": (".rec", {"binary_mode": True}),
    "arec": (".arec", {"binary_mode": True}),
    "fe": (".fe", {}),
    "ply-text": (".ply", {"binary_mode": False}),
    "ply-binary": (".ply", {"binary_mode": True}),
    "ply-colors": (".ply", {"binary_mode": True, "colors": True}),
    "obj": (".obj", {}),
    "3mf": (".3mf", {}),
}
NAMES = list(VARIANTS)


@pytest.fixture(scope="module")
def mesh_for_matrix() -> Mesh:
    """Sixteen closed cells in a medium (label 0)."""
    return iorec.load_rec(ASSETS / "binary.rec")


@pytest.fixture(scope="module")
def source_files(
    tmp_path_factory: pytest.TempPathFactory,
    mesh_for_matrix: Mesh,
) -> dict[str, Path]:
    """The reference mesh written once in each variant."""
    folder = tmp_path_factory.mktemp("sources")
    files = {}
    for name, (suffix, options) in VARIANTS.items():
        files[name] = folder / f"source_{name}{suffix}"
        iorec.save_mesh(files[name], *mesh_for_matrix, **options)
    return files


@pytest.mark.parametrize("source", NAMES)
def test_source_file_reproduces_the_mesh(
    source: str,
    source_files: dict[str, Path],
    mesh_for_matrix: Mesh,
    same_mesh: Callable[[Mesh, Mesh], bool],
) -> None:
    """Reference point of the matrix: each format alone keeps the mesh."""
    assert same_mesh(iorec.load_mesh(source_files[source]), mesh_for_matrix)


@pytest.mark.parametrize("target", NAMES)
@pytest.mark.parametrize("source", NAMES)
def test_conversion_matrix(
    source: str,
    target: str,
    tmp_path: Path,
    source_files: dict[str, Path],
    mesh_for_matrix: Mesh,
    same_mesh: Callable[[Mesh, Mesh], bool],
) -> None:
    """Source file -> convert -> target file -> load gives the reference mesh."""
    suffix, options = VARIANTS[target]
    destination = tmp_path / f"converted{suffix}"
    iorec.convert(source_files[source], destination, save_options=options)
    assert same_mesh(iorec.load_mesh(destination), mesh_for_matrix), (
        f"{source} -> {target}: labelled triangles differ"
    )
