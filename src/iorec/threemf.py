"""3MF files (.3mf) for 3D printing with Cura (Ultimaker), PrusaSlicer or Bambu Studio.

A 3MF object is a closed, outward-oriented surface, so iorec writes one object per
label but the exterior: the whole surface of that cell, a triangle between two cells
being written in both. The cells are parts of one assembly, so that a slicer moves
and drops them together. Each cell carries its iorec label, a colour and, optionally,
a Cura extruder number (`cura:extruder_nr`), to print chosen cells in the second
material of a dual-extrusion printer such as the Ultimaker S5.

Vertices are written exactly, in the frame of the mesh. The printed size is set by the
3MF unit and a scale factor in the build transform, which also puts the lowest corner
of the assembly at the origin of the build plate. Reading a file written by iorec
ignores that transform and gives back the mesh, up to the order of vertices and
triangles and the orientation convention of each triangle. Other 3MF files are read
with their transforms applied, each mesh object becoming one label (1, 2, ...).
"""

from __future__ import annotations

import warnings
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Dict, List, Mapping, Tuple, Union
from xml.etree import ElementTree

import numpy as np
from numpy.typing import ArrayLike, NDArray

from iorec._colors import label_color, to_hex
from iorec._topology import (
    _cell_shells,
    _merge_coincident_vertices,
    _merge_shells,
    _mesh_arrays,
    _open_half_edges,
    _resolve_mesh_path,
)
from iorec._validation import MeshFormatError, _check_triangle_mesh
from iorec.rec_appendix import _iorec_version

_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_CURA = "http://software.ultimaker.com/xml/cura/3mf/2015/10"
_IOREC = "https://src.koda.cnrs.fr/turlierlab/iorec"
_MODEL_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
_PRODUCTION_PATH = (
    "{http://schemas.microsoft.com/3dmanufacturing/production/2015/06}path"
)
_UNITS = ("micron", "millimeter", "centimeter", "inch", "foot", "meter")
_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" '
    'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="model" '
    'ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
    "</Types>\n"
)
_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    f'<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="{_MODEL_REL}"/>'
    "</Relationships>\n"
)

# A mesh object (vertices, triangles, iorec label or None), or a list of components.
_Mesh = Tuple[NDArray[np.float64], NDArray[np.int64], Union[int, None]]
_Components = List[Tuple[str, NDArray[np.float64]]]
_Objects = Dict[str, Union[_Mesh, _Components]]


def save_3mf(
    filename: str | Path,
    points: ArrayLike,
    triangles: ArrayLike,
    labels: ArrayLike,
    *,
    exterior_label: int = 0,
    unit: str = "millimeter",
    scale: float = 1.0,
    extruders: Mapping[int, int] | None = None,
) -> None:
    """Write the cells of a labelled mesh as the parts of a 3MF assembly.

    Args:
        filename: where to write the file (.3mf).
        points: vertex coordinates, shape (n, 3).
        triangles: vertex indices, shape (m, 3).
        labels: the two labels of each triangle, shape (m, 2).
        exterior_label: the label that is not printed.
        unit: 3MF length unit of the scaled coordinates, e.g. "millimeter", "micron".
        scale: printed size = coordinates * scale, in `unit`.
        extruders: {label: Cura extruder number}, 0 for the first extruder.
    """
    pts, tri, lab = _mesh_arrays(points, triangles, labels)
    if unit not in _UNITS:
        error = f"unit must be one of {', '.join(_UNITS)}, got {unit!r}."
        raise ValueError(error)
    if not (np.isfinite(scale) and scale > 0):
        error = f"scale must be a positive number, got {scale!r}."
        raise ValueError(error)
    shells = _cell_shells(tri, lab, exterior_label)
    _warn_open_shells(shells)
    model = "".join(
        _model_xml(pts, shells, exterior_label, unit, scale, dict(extruders or {})),
    )
    with zipfile.ZipFile(
        filename,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as archive:
        archive.writestr("[Content_Types].xml", _CONTENT_TYPES)
        archive.writestr("_rels/.rels", _RELS)
        archive.writestr("3D/3dmodel.model", model)


def _warn_open_shells(shells: list[tuple[int, NDArray[np.int64]]]) -> None:
    for label, shell in shells:
        open_edges = _open_half_edges(shell)
        if open_edges:
            message = (
                f"the surface of label {label} is not closed ({open_edges} open "
                "edges): a slicer may repair it, print it wrongly or reject it."
            )
            warnings.warn(message, UserWarning, stacklevel=3)


def _model_xml(
    points: NDArray[np.float64],
    shells: list[tuple[int, NDArray[np.int64]]],
    exterior_label: int,
    unit: str,
    scale: float,
    extruders: dict[int, int],
) -> Iterator[str]:
    """Yield the text of the 3D model part, piece by piece."""
    yield (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="{unit}" xml:lang="en-US" xmlns="{_CORE}" '
        f'xmlns:cura="{_CURA}" xmlns:iorec="{_IOREC}">\n'
        f'<metadata name="Application">iorec {_iorec_version()}</metadata>\n'
        f'<metadata name="iorec:exterior_label">{exterior_label}</metadata>\n'
        "<resources>\n"
        '<basematerials id="1">\n'
    )
    for label, _ in shells:
        color = to_hex(label_color(label))
        yield f'<base name="cell {label}" displaycolor="{color}"/>\n'
    yield "</basematerials>\n"
    for index, (label, shell) in enumerate(shells):
        yield from _object_xml(points, label, shell, index, extruders.get(label))
    assembly = len(shells) + 2
    yield f'<object id="{assembly}" type="model" name="cells">\n<components>\n'
    for index in range(len(shells)):
        yield f'<component objectid="{index + 2}"/>\n'
    yield "</components>\n</object>\n</resources>\n<build>\n"
    transform = _build_transform(points, shells, scale)
    yield f'<item objectid="{assembly}" transform="{transform}"/>\n</build>\n</model>\n'


def _object_xml(
    points: NDArray[np.float64],
    label: int,
    shell: NDArray[np.int64],
    index: int,
    extruder: int | None,
) -> Iterator[str]:
    """Yield one cell as a 3MF mesh object, with only the vertices it uses."""
    used, local = np.unique(shell, return_inverse=True)
    vertices = points[used]
    yield (
        f'<object id="{index + 2}" type="model" name="cell {label}" '
        f'pid="1" pindex="{index}">\n<metadatagroup>\n'
        f'<metadata name="iorec:label">{label}</metadata>\n'
    )
    if extruder is not None:
        cura = '<metadata name="cura:extruder_nr" preserve="1">'
        yield f"{cura}{int(extruder)}</metadata>\n"
    yield "</metadatagroup>\n<mesh>\n<vertices>\n"
    row = '<vertex x="%r" y="%r" z="%r"/>\n'
    yield (row * len(vertices)) % tuple(vertices.reshape(-1).tolist())
    yield "</vertices>\n<triangles>\n"
    row = '<triangle v1="%d" v2="%d" v3="%d"/>\n'
    yield (row * len(shell)) % tuple(local.reshape(-1).tolist())
    yield "</triangles>\n</mesh>\n</object>\n"


def _build_transform(
    points: NDArray[np.float64],
    shells: list[tuple[int, NDArray[np.int64]]],
    scale: float,
) -> str:
    """3MF row-major affine matrix: scale, then lowest corner at the origin."""
    if not shells:
        return "1 0 0 0 1 0 0 0 1 0 0 0"
    used = np.unique(np.concatenate([s for _, s in shells]))
    lowest = points[used].min(axis=0) if len(used) else np.zeros(3)
    shift = (-scale * lowest + 0.0).tolist()
    return " ".join(
        repr(float(x)) for x in [scale, 0, 0, 0, scale, 0, 0, 0, scale, *shift]
    )


def load_3mf(
    filename: str | Path,
    *,
    exterior_label: int | None = None,
    validate: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a 3MF file as one labelled mesh, merging the surfaces of its objects.

    Args:
        filename: a .3mf file.
        exterior_label: label of the side of cell surfaces facing no other cell.
            Defaults to the value iorec wrote in the file, or 0.
        validate: also check coordinates, indices and labels as `load_rec` does.

    Returns:
        (points, triangles, labels): float64 (n, 3), int64 (m, 3) and int64 (m, 2).
    """
    path = _resolve_mesh_path(filename, (".3mf",))
    root = _read_model(path)
    metadata = {
        m.get("name"): (m.text or "").strip() for m in root.iter(f"{{{_CORE}}}metadata")
    }
    written_by_iorec = "iorec:exterior_label" in metadata
    if exterior_label is None:
        exterior_label = int(metadata.get("iorec:exterior_label", 0))
    objects = _parse_objects(root, path)
    parts = list(
        _build_parts(root, objects, use_transforms=not written_by_iorec, source=path),
    )
    points, shells = _join_parts(parts)
    triangles = (
        np.concatenate([s for _, s in shells]) if shells else np.empty((0, 3), np.int64)
    )
    points, triangles = _merge_coincident_vertices(points, triangles)
    bounds = np.cumsum([0, *(len(s) for _, s in shells)])
    shells = [
        (lab, triangles[bounds[k] : bounds[k + 1]]) for k, (lab, _) in enumerate(shells)
    ]
    triangles, labels = _merge_shells(shells, exterior_label, path)
    if validate:
        _check_triangle_mesh(path, points, triangles, labels)
    return points, triangles, labels


def _read_model(path: Path) -> ElementTree.Element:
    """Parse the root 3D model part named by the package relationships."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            target = "3D/3dmodel.model"
            if "_rels/.rels" in names:
                # The XML parser expands no external entity and bounds entity
                # expansion, so a local 3MF package is parsed with the standard library.
                rels = ElementTree.fromstring(archive.read("_rels/.rels"))  # noqa: S314
                for rel in rels:
                    if rel.get("Type") == _MODEL_REL:
                        target = (rel.get("Target") or target).lstrip("/")
            return ElementTree.fromstring(archive.read(target))  # noqa: S314
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        error = f"{path}: not a readable 3MF package: {exc}"
        raise MeshFormatError(error) from exc


def _parse_objects(root: ElementTree.Element, source: Path) -> _Objects:
    objects: _Objects = {}
    for obj in root.iter(f"{{{_CORE}}}object"):
        mesh = obj.find(f"{{{_CORE}}}mesh")
        if mesh is not None:
            objects[obj.get("id", "")] = _parse_mesh(obj, mesh)
            continue
        components = []
        for component in obj.iter(f"{{{_CORE}}}component"):
            if component.get(_PRODUCTION_PATH):
                error = f"{source}: components in other model files are not supported."
                raise MeshFormatError(error)
            matrix = _matrix(component.get("transform"))
            components.append((component.get("objectid", ""), matrix))
        objects[obj.get("id", "")] = components
    return objects


def _parse_mesh(obj: ElementTree.Element, mesh: ElementTree.Element) -> _Mesh:
    vertices = np.asarray(
        [[v.get("x"), v.get("y"), v.get("z")] for v in mesh.iter(f"{{{_CORE}}}vertex")],
        dtype=np.float64,
    ).reshape(-1, 3)
    triangles = np.asarray(
        [
            [t.get("v1"), t.get("v2"), t.get("v3")]
            for t in mesh.iter(f"{{{_CORE}}}triangle")
        ],
        dtype=np.int64,
    ).reshape(-1, 3)
    label = None
    for meta in obj.iter(f"{{{_CORE}}}metadata"):
        if meta.get("name") == "iorec:label":
            label = int((meta.text or "").strip())
    return vertices, triangles, label


def _matrix(transform: str | None) -> NDArray[np.float64]:
    """3MF transform string as a 4x3 matrix acting on row vectors [x y z 1]."""
    if not transform:
        return np.vstack([np.eye(3), np.zeros(3)])
    return np.asarray(transform.split(), dtype=np.float64).reshape(4, 3)


def _build_parts(
    root: ElementTree.Element,
    objects: _Objects,
    *,
    use_transforms: bool,
    source: Path,
) -> Iterator[_Mesh]:
    """Yield every mesh placed on the build plate, in world coordinates if asked."""
    for item in root.iter(f"{{{_CORE}}}item"):
        matrix = _matrix(item.get("transform"))
        yield from _expand(
            item.get("objectid", ""),
            matrix,
            objects,
            use_transforms,
            source,
            0,
        )


def _expand(
    object_id: str,
    matrix: NDArray[np.float64],
    objects: _Objects,
    use_transforms: bool,
    source: Path,
    depth: int,
) -> Iterator[_Mesh]:
    entry = objects.get(object_id)
    if entry is None or depth > 32:
        error = f"{source}: missing or cyclic object reference {object_id!r}."
        raise MeshFormatError(error)
    if isinstance(entry, tuple):
        vertices, triangles, label = entry
        if use_transforms:
            vertices = vertices @ matrix[:3] + matrix[3]
        yield vertices, triangles, label
        return
    for child, child_matrix in entry:
        combined = np.vstack(
            [child_matrix[:3] @ matrix[:3], child_matrix[3] @ matrix[:3] + matrix[3]],
        )
        yield from _expand(child, combined, objects, use_transforms, source, depth + 1)


def _join_parts(
    parts: list[_Mesh],
) -> tuple[NDArray[np.float64], list[tuple[int, NDArray[np.int64]]]]:
    """Stack the vertices of all parts; label each part (1, 2, ... when unlabelled)."""
    offsets = np.cumsum([0, *(len(v) for v, _, _ in parts)])
    points = np.concatenate([v for v, _, _ in parts]) if parts else np.empty((0, 3))
    shells = [
        (k + 1 if label is None else label, triangles + offsets[k])
        for k, (_, triangles, label) in enumerate(parts)
    ]
    return points, shells
