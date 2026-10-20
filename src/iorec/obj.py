"""Wavefront OBJ files (.obj, with a .mtl of colours) for Blender and other 3D tools.

The mesh is written as one OBJ object per group of triangles, each with a material of
the same name in a .mtl file beside it, so that Blender imports one coloured object
per group. Two groupings are offered:

- `interface` (default): object `interface_<a>_<b>` holds the triangles between labels
  a < b, oriented so that their normals point from a to b. Each triangle appears once.
- `cell`: object `cell_<k>` holds the surface of label k, oriented outward. A triangle
  between two cells appears in both.

Coordinates are written exactly. Reading back gives the same mesh up to the order of
triangles and the orientation convention: a triangle with labels (b, a), b > a, comes
back reversed with labels (a, b). Objects with other names are read as the surfaces of
cells 1, 2, ... in file order, the remaining side being the exterior.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from iorec._colors import interface_color, label_color
from iorec._topology import (
    _cell_shells,
    _fan,
    _merge_coincident_vertices,
    _merge_shells,
    _mesh_arrays,
    _resolve_mesh_path,
    _split_shells,
)
from iorec._validation import MeshFormatError, _check_triangle_mesh
from iorec.rec_appendix import _iorec_version

_INTERFACE = re.compile(r"interface_(\d+)_(\d+)(?:\.\d+)?")
_CELL = re.compile(r"cell_(\d+)(?:\.\d+)?")
_EXTERIOR_COMMENT = re.compile(r"#\s*iorec exterior_label\s+(\d+)")
_SUFFIXES = (".obj",)


def save_obj(
    filename: str | Path,
    points: ArrayLike,
    triangles: ArrayLike,
    labels: ArrayLike,
    *,
    group_by: str = "interface",
    exterior_label: int = 0,
    write_mtl: bool = True,
) -> None:
    """Write a labelled triangle mesh as an OBJ file, and its colours as a .mtl file.

    Args:
        filename: where to write the file (.obj); the .mtl gets the same name.
        points: vertex coordinates, shape (n, 3).
        triangles: vertex indices, shape (m, 3).
        labels: the two labels of each triangle, shape (m, 2).
        group_by: "interface" (one object per pair of labels) or "cell" (one closed
            surface per label but the exterior).
        exterior_label: the label without a cell object, coloured as its neighbour.
        write_mtl: write the .mtl file of colours.
    """
    pts, tri, lab = _mesh_arrays(points, triangles, labels)
    groups = _groups(tri, lab, group_by, exterior_label)
    path = Path(filename)
    mtl_path = path.with_suffix(".mtl")
    lines = [
        f"# written by iorec {_iorec_version()}",
        f"# iorec group_by {group_by}",
        f"# iorec exterior_label {exterior_label}",
    ]
    if write_mtl:
        lines.append(f"mtllib {mtl_path.name}")
    text = "\n".join(lines) + "\n"
    text += ("v %r %r %r\n" * len(pts)) % tuple(pts.reshape(-1).tolist())
    for name, _, group_triangles in groups:
        faces = (group_triangles + 1).reshape(-1).tolist()
        text += f"o {name}\nusemtl {name}\n"
        text += ("f %d %d %d\n" * len(group_triangles)) % tuple(faces)
    path.write_text(text)
    if write_mtl:
        mtl_path.write_text(_mtl_text(groups))


def _groups(
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    group_by: str,
    exterior_label: int,
) -> list[tuple[str, tuple[float, float, float], NDArray[np.int64]]]:
    """(object name, colour, triangles) for each OBJ object."""
    if group_by == "cell":
        return [
            (f"cell_{label}", label_color(label), shell)
            for label, shell in _cell_shells(triangles, labels, exterior_label)
        ]
    if group_by != "interface":
        error = f"group_by must be 'interface' or 'cell', got {group_by!r}."
        raise ValueError(error)
    flip = labels[:, 0] > labels[:, 1]
    oriented = np.where(flip[:, None], triangles[:, ::-1], triangles)
    pairs, group = np.unique(np.sort(labels, axis=1), axis=0, return_inverse=True)
    order = np.argsort(group.reshape(-1), kind="stable")
    bounds = np.searchsorted(group.reshape(-1)[order], np.arange(len(pairs) + 1))
    groups = []
    for k, (a, b) in enumerate(pairs.tolist()):
        members = order[bounds[k] : bounds[k + 1]]
        color = interface_color(a, b, exterior_label)
        groups.append((f"interface_{a}_{b}", color, oriented[members]))
    return groups


def _mtl_text(
    groups: list[tuple[str, tuple[float, float, float], NDArray[np.int64]]],
) -> str:
    blocks = [f"# written by iorec {_iorec_version()}"]
    for name, (r, g, b), _ in groups:
        blocks.append(f"newmtl {name}\nKd {r:.4f} {g:.4f} {b:.4f}\nd 1.0\nillum 1")
    return "\n\n".join(blocks) + "\n"


def load_obj(
    filename: str | Path,
    *,
    exterior_label: int | None = None,
    merge_vertices: bool = False,
    validate: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read an OBJ file as points, triangles and labels.

    Args:
        filename: a .obj file; materials and texture or normal indices are ignored.
        exterior_label: label of the side of cell surfaces facing no other cell.
            Defaults to the value iorec wrote in the file, or 0.
        merge_vertices: merge vertices with equal coordinates, for files in which
            each object has its own copy of shared vertices (Blender exports).
        validate: also check coordinates, indices and labels as `load_rec` does.

    Returns:
        (points, triangles, labels): float64 (n, 3), int64 (m, 3) and int64 (m, 2).
    """
    path = _resolve_mesh_path(filename, _SUFFIXES)
    text = path.read_text(encoding="utf-8", errors="replace")
    points, groups = _parse_obj(text, path)
    triangles = np.concatenate(
        [np.asarray(t, np.int64).reshape(-1, 3) for _, t in groups],
    )
    if merge_vertices:
        points, triangles = _merge_coincident_vertices(points, triangles)
    if exterior_label is None:
        exterior_label = _exterior_label_of(text)
    triangles, labels = _group_labels(groups, triangles, exterior_label, path)
    if validate:
        _check_triangle_mesh(path, points, triangles, labels)
    return points, triangles, labels


def _stored_exterior_label(filename: str | Path) -> int:
    """The exterior label iorec wrote in an OBJ file, or 0."""
    path = _resolve_mesh_path(filename, _SUFFIXES)
    return _exterior_label_of(path.read_text(encoding="utf-8", errors="replace"))


def _exterior_label_of(text: str) -> int:
    found = _EXTERIOR_COMMENT.search(text)
    return int(found.group(1)) if found else 0


def _parse_obj(
    text: str,
    source: Path,
) -> tuple[NDArray[np.float64], list[tuple[str, list[list[int]]]]]:
    """Vertices, and the triangles of each named group in order of appearance."""
    vertices: list[list[float]] = []
    groups: dict[str, list[list[int]]] = {}
    name = ""
    for number, line in enumerate(text.splitlines(), start=1):
        words = line.split()
        if not words:
            continue
        if words[0] == "v":
            vertices.append([float(w) for w in words[1:4]])
        elif words[0] in ("o", "g") or (words[0] == "usemtl" and not name):
            name = " ".join(words[1:])
        elif words[0] == "f":
            polygon = [
                _vertex_index(w, len(vertices), source, number) for w in words[1:]
            ]
            groups.setdefault(name, []).extend(_fan(polygon))
    if not groups:
        groups[""] = []
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    return points, list(groups.items())


def _vertex_index(word: str, n_vertices: int, source: Path, number: int) -> int:
    """0-based vertex index of a face corner `v`, `v/vt`, `v//vn` or `v/vt/vn`."""
    index = int(word.split("/")[0])
    if index == 0 or (index < 0 and -index > n_vertices):
        error = f"{source}:{number}: invalid vertex index {index}."
        raise MeshFormatError(error)
    return index - 1 if index > 0 else n_vertices + index


def _group_labels(
    groups: list[tuple[str, list[list[int]]]],
    triangles: NDArray[np.int64],
    exterior_label: int,
    source: Path,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Labels from iorec interface names, or by merging the surfaces of cells."""
    sizes = [len(t) for _, t in groups]
    interfaces = [_INTERFACE.fullmatch(name) for name, _ in groups]
    if all(interfaces):
        pairs = [(int(m.group(1)), int(m.group(2))) for m in interfaces if m]
        labels = np.repeat(np.asarray(pairs, np.int64).reshape(-1, 2), sizes, axis=0)
        return triangles, labels
    cells = [_CELL.fullmatch(name) for name, _ in groups]
    if all(cells):
        cell_labels = [int(m.group(1)) for m in cells if m]
    else:
        cell_labels = list(range(1, len(groups) + 1))
    shells = _split_shells(cell_labels, sizes, triangles)
    return _merge_shells(shells, exterior_label, source)
