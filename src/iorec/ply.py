"""PLY files (.ply): the labelled nonmanifold mesh for MeshLab, ParaView or Blender.

iorec writes a `vertex` element (x, y, z as double) and a `face` element with the
vertex indices and two int properties, `label0` and `label1`: the face normal
(right-hand rule) points from `label0` to `label1`, as in .rec files. Values are
stored exactly, in binary (little-endian) or text, so a .rec -> .ply -> .rec round trip
gives back the same arrays. Optional `red green blue` face colours show the interfaces
in viewers that read face colours (MeshLab, ParaView).

The reader takes any text or binary PLY with `vertex` and `face` elements. Polygons
are split into triangles around their first vertex. Faces without `label0`/`label1`
get labels (1, 0): one region inside, the exterior outside.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from iorec._colors import interface_color, to_bytes
from iorec._topology import _fan, _mesh_arrays, _resolve_mesh_path
from iorec._validation import MeshFormatError, _check_mesh
from iorec.rec_appendix import _iorec_version

_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}
_ENDIAN = {"binary_little_endian": "<", "binary_big_endian": ">", "ascii": None}
_INDEX_NAMES = ("vertex_indices", "vertex_index")

# One property: (name, value type, count type for a list property or None).
_Property = Tuple[str, str, Optional[str]]
_Element = Tuple[str, int, List[_Property]]


def save_ply(
    filename: str | Path,
    points: ArrayLike,
    triangles: ArrayLike,
    labels: ArrayLike,
    *,
    binary_mode: bool = True,
    colors: bool = False,
    exterior_label: int = 0,
) -> None:
    """Write a labelled triangle mesh as a PLY file.

    Args:
        filename: where to write the file (.ply).
        points: vertex coordinates, shape (n, 3), written as double.
        triangles: vertex indices, shape (m, 3), written as int.
        labels: the two labels of each triangle, shape (m, 2), written as int.
        binary_mode: binary little-endian (default) or text.
        colors: also write a `red green blue` colour per face, from its labels.
        exterior_label: the label whose interfaces take the colour of the other side.
    """
    pts, tri, lab = _mesh_arrays(points, triangles, labels)
    if max(len(pts), int(np.abs(lab).max(initial=0))) > np.iinfo(np.int32).max:
        error = "PLY int properties hold at most 2**31 - 1 vertices and labels."
        raise ValueError(error)
    face_colors = _face_colors(lab, exterior_label) if colors else None
    header = _header(len(pts), len(tri), binary_mode=binary_mode, colors=colors)
    with Path(filename).open("wb") as file:
        file.write(header.encode("ascii"))
        if binary_mode:
            file.write(_binary_body(pts, tri, lab, face_colors))
        else:
            file.write(_text_body(pts, tri, lab, face_colors).encode("ascii"))


def _header(n_points: int, n_faces: int, *, binary_mode: bool, colors: bool) -> str:
    lines = [
        "ply",
        "format binary_little_endian 1.0" if binary_mode else "format ascii 1.0",
        f"comment written by iorec {_iorec_version()}",
        "comment face normal (right-hand rule) points from label0 to label1",
        f"element vertex {n_points}",
        "property double x",
        "property double y",
        "property double z",
        f"element face {n_faces}",
        "property list uchar int vertex_indices",
        "property int label0",
        "property int label1",
    ]
    if colors:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    return "\n".join([*lines, "end_header"]) + "\n"


def _face_colors(labels: NDArray[np.int64], exterior_label: int) -> NDArray[np.uint8]:
    pairs, inverse = np.unique(labels, axis=0, return_inverse=True)
    table = np.asarray(
        [to_bytes(interface_color(a, b, exterior_label)) for a, b in pairs.tolist()],
        dtype=np.uint8,
    ).reshape(-1, 3)
    return table[inverse.reshape(-1)]


def _binary_body(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    colors: NDArray[np.uint8] | None,
) -> bytes:
    fields: list[tuple[Any, ...]] = [
        ("count", "u1"),
        ("vertices", "<i4", (3,)),
        ("label0", "<i4"),
        ("label1", "<i4"),
    ]
    if colors is not None:
        fields.append(("rgb", "u1", (3,)))
    faces = np.empty(len(triangles), dtype=np.dtype(fields))
    faces["count"] = 3
    faces["vertices"] = triangles
    faces["label0"] = labels[:, 0]
    faces["label1"] = labels[:, 1]
    if colors is not None:
        faces["rgb"] = colors
    return np.ascontiguousarray(points, dtype="<f8").tobytes() + faces.tobytes()


def _text_body(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    colors: NDArray[np.uint8] | None,
) -> str:
    """Rows of values written with `repr`, which reads back to the same float64."""
    columns = [np.full((len(triangles), 1), 3), triangles, labels]
    if colors is not None:
        columns.append(colors.astype(np.int64))
    faces = np.hstack(columns) if len(triangles) else np.empty((0, 6), np.int64)
    point_rows = ("%r %r %r\n" * len(points)) % tuple(points.reshape(-1).tolist())
    face_row = " ".join(["%d"] * faces.shape[1]) + "\n"
    face_rows = (face_row * len(faces)) % tuple(faces.reshape(-1).tolist())
    return point_rows + face_rows


def load_ply(
    filename: str | Path,
    *,
    validate: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a PLY file as points, triangles and labels.

    Args:
        filename: a .ply file, text or binary.
        validate: also check coordinates, indices and labels as `load_rec` does.

    Returns:
        (points, triangles, labels): float64 (n, 3), int64 (m, 3) and int64 (m, 2).

    Raises:
        MeshFormatError: if the file is not a PLY mesh or is truncated.
    """
    path = _resolve_mesh_path(filename, (".ply",))
    data = path.read_bytes()
    form, elements, offset = _parse_header(data, path)
    values = _read_elements(data, offset, elements, form, path)
    points, triangles, labels = _mesh_from_values(values, path)
    if validate:
        counts = (len(points), len(triangles))
        _check_mesh(
            path,
            points,
            triangles,
            labels,
            3,
            counts,
            error_type=MeshFormatError,
        )
    return points, triangles, labels


def _parse_header(data: bytes, source: Path) -> tuple[str, list[_Element], int]:
    """Return the format, the declared elements and the offset of the data."""
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        error = f"{source}: not a PLY file (no 'ply' ... 'end_header' header)."
        raise MeshFormatError(error)
    offset = data.index(b"\n", end) + 1
    form = ""
    elements: list[_Element] = []
    for line in data[:end].decode("ascii", errors="replace").splitlines()[1:]:
        words = line.split()
        if not words or words[0] in ("comment", "obj_info"):
            continue
        if words[0] == "format":
            form = words[1]
        elif words[0] == "element":
            elements.append((words[1], int(words[2]), []))
        elif words[0] == "property" and elements:
            elements[-1][2].append(_parse_property(words, source))
    if form not in _ENDIAN:
        error = f"{source}: unknown PLY format {form!r}."
        raise MeshFormatError(error)
    return form, elements, offset


def _parse_property(words: list[str], source: Path) -> _Property:
    if words[1] == "list" and len(words) == 5:
        count_type, value_type, name = words[2:5]
    elif len(words) == 3:
        count_type, (value_type, name) = None, words[1:3]
    else:
        count_type = value_type = name = ""
    types_known = value_type in _TYPES and (count_type is None or count_type in _TYPES)
    if not types_known:
        error = f"{source}: cannot read the PLY property line {' '.join(words)!r}."
        raise MeshFormatError(error)
    return name, _TYPES[value_type], None if count_type is None else _TYPES[count_type]


def _read_elements(
    data: bytes,
    offset: int,
    elements: list[_Element],
    form: str,
    source: Path,
) -> dict[str, dict[str, Any]]:
    """Values of the vertex and face elements, by element and property name."""
    endian = _ENDIAN[form]
    tokens = data[offset:].split() if endian is None else []
    position = 0 if endian is None else offset
    values: dict[str, dict[str, Any]] = {}
    for name, count, properties in elements:
        if name in values or ("vertex" in values and "face" in values):
            break
        try:
            if endian is None:
                element, position = _text_element(tokens, position, count, properties)
            else:
                element, position = _binary_element(
                    data,
                    position,
                    count,
                    properties,
                    endian,
                )
        except (ValueError, IndexError) as exc:
            error = f"{source}: truncated or malformed PLY element {name!r}: {exc}"
            raise MeshFormatError(error) from exc
        values[name] = element
    return values


def _binary_element(
    data: bytes,
    offset: int,
    count: int,
    properties: list[_Property],
    endian: str,
) -> tuple[dict[str, Any], int]:
    """Read one binary element; lists of equal length are read in one block."""
    if count == 0:
        return {name: np.empty(0) for name, _, _ in properties}, offset
    lengths = _first_list_lengths(data, offset, properties, endian)
    fields: list[tuple[Any, ...]] = []
    for (name, value_type, count_type), length in zip(properties, lengths):
        if count_type is None:
            fields.append((name, endian + value_type))
        else:
            fields.append((f"{name}#count", endian + count_type))
            fields.append((name, endian + value_type, (length,)))
    dtype = np.dtype(fields)
    if count and offset + dtype.itemsize * count > len(data):
        error = f"{count} records of {dtype.itemsize} bytes do not fit in the file"
        raise ValueError(error)
    block = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
    uniform = all(
        (block[f"{name}#count"] == length).all()
        for (name, _, count_type), length in zip(properties, lengths)
        if count_type is not None
    )
    if not uniform:
        return _binary_element_by_record(data, offset, count, properties, endian)
    element = {name: block[name] for name, _, _ in properties}
    return element, offset + dtype.itemsize * count


def _first_list_lengths(
    data: bytes,
    offset: int,
    properties: list[_Property],
    endian: str,
) -> list[int]:
    """Length of each list property in the first record (0 for scalars)."""
    lengths = []
    for _, value_type, count_type in properties:
        if count_type is None:
            lengths.append(0)
            offset += np.dtype(value_type).itemsize
            continue
        length = int(
            np.frombuffer(data, endian + count_type, count=1, offset=offset)[0],
        )
        lengths.append(length)
        offset += np.dtype(count_type).itemsize + length * np.dtype(value_type).itemsize
    return lengths


def _binary_element_by_record(
    data: bytes,
    offset: int,
    count: int,
    properties: list[_Property],
    endian: str,
) -> tuple[dict[str, Any], int]:
    """Read one binary element record by record, for lists of different lengths."""
    element: dict[str, list[Any]] = {name: [] for name, _, _ in properties}
    for _ in range(count):
        for name, value_type, count_type in properties:
            if count_type is None:
                value = np.frombuffer(data, endian + value_type, count=1, offset=offset)
                element[name].append(value[0])
                offset += value.itemsize
                continue
            length = int(np.frombuffer(data, endian + count_type, 1, offset)[0])
            offset += np.dtype(count_type).itemsize
            items = np.frombuffer(data, endian + value_type, length, offset)
            element[name].append(items.tolist())
            offset += items.nbytes
    return {name: _as_column(column) for name, column in element.items()}, offset


def _text_element(
    tokens: list[bytes],
    position: int,
    count: int,
    properties: list[_Property],
) -> tuple[dict[str, Any], int]:
    """Read one text element; records of equal shape are read in one block."""
    if count == 0:
        return {name: np.empty(0) for name, _, _ in properties}, position
    lengths, width = [], 0
    for _, _, count_type in properties:
        length = 0 if count_type is None else int(tokens[position + width])
        lengths.append(length)
        width += 1 if count_type is None else 1 + length
    block = np.asarray(tokens[position : position + width * count], dtype=np.float64)
    if block.size != width * count:
        return _text_element_by_record(tokens, position, count, properties)
    table = block.reshape(count, width)
    element: dict[str, Any] = {}
    column = 0
    for (name, _, count_type), length in zip(properties, lengths):
        if count_type is None:
            element[name] = table[:, column]
            column += 1
            continue
        if not (table[:, column] == length).all():
            return _text_element_by_record(tokens, position, count, properties)
        element[name] = table[:, column + 1 : column + 1 + length].astype(np.int64)
        column += 1 + length
    return element, position + width * count


def _text_element_by_record(
    tokens: list[bytes],
    position: int,
    count: int,
    properties: list[_Property],
) -> tuple[dict[str, Any], int]:
    """Read one text element record by record, for lists of different lengths."""
    columns: dict[str, list[Any]] = {name: [] for name, _, _ in properties}
    for _ in range(count):
        for name, _, count_type in properties:
            if count_type is None:
                columns[name].append(float(tokens[position]))
                position += 1
                continue
            length = int(tokens[position])
            items = tokens[position + 1 : position + 1 + length]
            if len(items) != length:
                error = "the last record is incomplete"
                raise ValueError(error)
            columns[name].append([int(t) for t in items])
            position += 1 + length
    return {name: _as_column(column) for name, column in columns.items()}, position


def _as_column(column: list[Any]) -> NDArray[Any] | list[Any]:
    """An array when all records have the same shape, the list itself otherwise."""
    if column and isinstance(column[0], list) and len({len(c) for c in column}) > 1:
        return column
    return np.asarray(column)


def _mesh_from_values(
    values: dict[str, dict[str, Any]],
    source: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    vertex, face = values.get("vertex", {}), values.get("face", {})
    if not all(axis in vertex for axis in "xyz"):
        error = f"{source}: the PLY file has no vertex x, y, z properties."
        raise MeshFormatError(error)
    points = np.column_stack([np.asarray(vertex[a], dtype=np.float64) for a in "xyz"])
    index_name = next((n for n in _INDEX_NAMES if n in face), None)
    if index_name is None:
        return points, np.empty((0, 3), np.int64), np.empty((0, 2), np.int64)
    polygons = face[index_name]
    if "label0" in face and "label1" in face:
        labels = np.column_stack([face["label0"], face["label1"]]).astype(np.int64)
    else:
        labels = np.tile(np.array([1, 0], dtype=np.int64), (len(polygons), 1))
    if (
        isinstance(polygons, np.ndarray)
        and polygons.ndim == 2
        and polygons.shape[1] == 3
    ):
        return points, polygons.astype(np.int64), labels
    return (points, *_split_polygons(polygons, labels, source))


def _split_polygons(
    polygons: NDArray[np.int64] | list[list[int]],
    labels: NDArray[np.int64],
    source: Path,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    triangles: list[list[int]] = []
    rows: list[int] = []
    for k, polygon in enumerate(polygons):
        if len(polygon) < 3:
            error = f"{source}: face {k} has fewer than three vertices."
            raise MeshFormatError(error)
        pieces = _fan([int(v) for v in polygon])
        triangles.extend(pieces)
        rows.extend([k] * len(pieces))
    tri = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    return tri, labels[np.asarray(rows, dtype=np.intp)]
