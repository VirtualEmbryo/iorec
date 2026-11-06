# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2023-2026 the iorec authors
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

import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from iorec import obj, ply, surface_evolver, threemf
from iorec._utils import _SUFFIXES as _REC_SUFFIXES
from iorec.rec_3d import load_rec, load_rec_with_metadata, save_rec

_Mesh = tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]


@dataclass(frozen=True)
class _Format:
    suffixes: tuple[str, ...]
    load: Callable[..., Any]
    save: Callable[..., None]
    write_suffixes: tuple[str, ...]
    stored_exterior_label: Callable[[str | Path], int] | None = None


_FORMATS = (
    _Format(_REC_SUFFIXES, load_rec, save_rec, _REC_SUFFIXES),
    _Format(
        surface_evolver._SUFFIXES,
        surface_evolver.load_fe,
        surface_evolver.save_fe,
        surface_evolver._SUFFIXES[:1],
    ),
    _Format(ply._SUFFIXES, ply.load_ply, ply.save_ply, ply._SUFFIXES),
    _Format(
        obj._SUFFIXES,
        obj.load_obj,
        obj.save_obj,
        obj._SUFFIXES,
        obj._stored_exterior_label,
    ),
    _Format(
        threemf._SUFFIXES,
        threemf.load_3mf,
        threemf.save_3mf,
        threemf._SUFFIXES,
        threemf._stored_exterior_label,
    ),
)
_READERS = {s: f for f in _FORMATS for s in f.suffixes}
_WRITERS = {s: f for f in _FORMATS for s in f.write_suffixes}


def _accepted_options(filename: str | Path, action: str) -> frozenset[str]:
    """Keyword arguments of the reader (`action="read"`) or writer of a suffix."""
    table = _READERS if action == "read" else _WRITERS
    format_ = table[_suffix(filename, table, action)]
    function = format_.load if action == "read" else format_.save
    return frozenset(inspect.signature(function).parameters)


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
    mesh: _Mesh = _READERS[_suffix(filename, _READERS, "read")].load(
        filename,
        **options,
    )
    return mesh


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
    _WRITERS[_suffix(filename, _WRITERS, "write")].save(
        filename,
        points,
        triangles,
        labels,
        **options,
    )


def convert(
    source: str | Path,
    target: str | Path,
    *,
    load_options: dict[str, Any] | None = None,
    save_options: dict[str, Any] | None = None,
) -> None:
    """Convert a mesh file to the format of the target suffix.

    The exterior label (`exterior_label` in the options, else the one stored in an
    .obj or .3mf source) is passed to the reader and to the writer when they take it.
    Extras of the source are kept when the target is of the same kind and dropped,
    with a warning, otherwise: the metadata appendix of a .rec/.arec file, the facet
    tensions and body volumes of a Surface Evolver file.
    """
    load_options = dict(load_options or {})
    save_options = dict(save_options or {})
    source_suffix = _suffix(source, _READERS, "read")
    target_suffix = _suffix(target, _WRITERS, "write")
    exterior = load_options.get("exterior_label")
    stored = _READERS[source_suffix].stored_exterior_label
    if exterior is None and stored is not None:
        exterior = stored(source)
    if exterior is not None:
        if "exterior_label" in _accepted_options(source, "read"):
            load_options["exterior_label"] = exterior
        if "exterior_label" in _accepted_options(target, "write"):
            save_options.setdefault("exterior_label", exterior)
    *mesh, extras, kind = _load_with_extras(source, source_suffix, load_options)
    if extras and _kind_of(target_suffix) == kind:
        for key, value in extras.items():
            save_options.setdefault(key, value)
    elif extras:
        message = f"not written to {target}: the {kind} of {source}."
        warnings.warn(message, UserWarning, stacklevel=2)
    save_mesh(target, *mesh, **save_options)


def _kind_of(suffix: str) -> str:
    if suffix in _REC_SUFFIXES:
        return "metadata appendix"
    if suffix in surface_evolver._SUFFIXES:
        return "facet tensions and body volumes"
    return ""


def _load_with_extras(
    source: str | Path,
    suffix: str,
    options: dict[str, Any],
) -> tuple[Any, ...]:
    """Mesh, writer options holding the extras of the source, and their kind."""
    kind = _kind_of(suffix)
    if suffix in _REC_SUFFIXES:
        *mesh, metadata = load_rec_with_metadata(source, **options)
        return (*mesh, {} if metadata is None else {"metadata": metadata}, kind)
    if suffix in surface_evolver._SUFFIXES:
        *mesh, attributes = surface_evolver.load_fe_with_attributes(source, **options)
        extras = {}
        if np.any(attributes["tension"] != 1.0):
            extras["tensions"] = attributes["tension"]
        if attributes["volume"]:
            extras["volumes"] = attributes["volume"]
        return (*mesh, extras, kind)
    return (*load_mesh(source, **options), {}, kind)
