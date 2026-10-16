"""Read, write and convert labelled meshes in any supported format, chosen by suffix.

======================  =========================  ================================
suffix                  format                     typical use
======================  =========================  ================================
``.rec``, ``.arec``     Delaunay-Watershed / MTT   deltamic, simba, exact storage
``.fe`` (``.dmp`` in)   Surface Evolver            energy minimisation
``.ply``                PLY                        exact exchange, MeshLab, ParaView
``.obj`` (+ ``.mtl``)   Wavefront OBJ              Blender, rendering
``.3mf``                3MF                        3D printing (Cura, PrusaSlicer)
======================  =========================  ================================

Every reader returns ``(points, triangles, labels)`` and every writer takes them, with
the convention of .rec files: the normal of triangle ``i`` (right-hand rule) points
from ``labels[i, 0]`` to ``labels[i, 1]``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from iorec.obj import load_obj, save_obj
from iorec.ply import load_ply, save_ply
from iorec.rec_3d import load_rec, load_rec_with_metadata, save_rec
from iorec.surface_evolver import load_fe, save_fe
from iorec.threemf import load_3mf, save_3mf

_Mesh = Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]
_READERS: dict[str, Callable[..., Any]] = {
    ".rec": load_rec,
    ".arec": load_rec,
    ".fe": load_fe,
    ".dmp": load_fe,
    ".ply": load_ply,
    ".obj": load_obj,
    ".3mf": load_3mf,
}
_WRITERS: dict[str, Callable[..., None]] = {
    ".rec": save_rec,
    ".arec": save_rec,
    ".fe": save_fe,
    ".ply": save_ply,
    ".obj": save_obj,
    ".3mf": save_3mf,
}
_REC = (".rec", ".arec")


def _suffix(filename: str | Path, table: dict[str, Any], action: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in table:
        error = (
            f"cannot {action} {filename}: the suffix must be one of "
            f"{', '.join(sorted(table))}."
        )
        raise ValueError(error)
    return suffix


def load_mesh(filename: str | Path, **options: Any) -> _Mesh:  # noqa: ANN401
    """Read a mesh with the reader of its suffix (`load_rec`, `load_fe`, ...).

    Args:
        filename: the mesh file.
        **options: keyword arguments of that reader, e.g. `validate=True`.

    Returns:
        (points, triangles, labels): float64 (n, 3), int64 (m, 3) and int64 (m, 2).
    """
    reader = _READERS[_suffix(filename, _READERS, "read")]
    points, triangles, labels = reader(filename, **options)
    return points, triangles, labels


def save_mesh(
    filename: str | Path,
    points: ArrayLike,
    triangles: ArrayLike,
    labels: ArrayLike,
    **options: Any,  # noqa: ANN401
) -> None:
    """Write a mesh with the writer of its suffix (`save_rec`, `save_fe`, ...).

    Args:
        filename: the mesh file to write.
        points: vertex coordinates, shape (n, 3).
        triangles: vertex indices, shape (m, 3).
        labels: the two labels of each triangle, shape (m, 2).
        **options: keyword arguments of that writer, e.g. `scale=2.0` for .3mf.
    """
    writer = _WRITERS[_suffix(filename, _WRITERS, "write")]
    writer(filename, points, triangles, labels, **options)


def convert(
    source: str | Path,
    target: str | Path,
    *,
    load_options: dict[str, Any] | None = None,
    save_options: dict[str, Any] | None = None,
) -> None:
    """Convert a mesh file to the format of the target suffix.

    The metadata appendix of a .rec/.arec source is kept when the target is .rec or
    .arec too; for other targets it is dropped, with a warning.
    """
    load_options = dict(load_options or {})
    save_options = dict(save_options or {})
    source_suffix = _suffix(source, _READERS, "read")
    target_suffix = _suffix(target, _WRITERS, "write")
    if source_suffix not in _REC:
        save_mesh(target, *load_mesh(source, **load_options), **save_options)
        return
    *mesh, metadata = load_rec_with_metadata(source, **load_options)
    if metadata is not None and target_suffix in _REC:
        save_options.setdefault("metadata", metadata)
    elif metadata is not None:
        message = f"the metadata appendix of {source} is not written to {target}."
        warnings.warn(message, UserWarning, stacklevel=2)
    save_mesh(target, *mesh, **save_options)
