"""Surface Evolver datafiles (.fe) and dumps (.dmp), 3D soapfilm linear model.

A Surface Evolver body is a material region: body ``k`` lists its boundary facets,
signed ``+`` when the facet normal (right-hand rule on its edge loop) points out of
the body. iorec labels hold the same information per triangle, whose normal points
from ``labels[:, 0]`` to ``labels[:, 1]``. iorec writes body ``k`` for label ``k`` and
no body for the exterior label (0 by default); on reading, a facet side without a body
gets the exterior label.

The reader takes the datafiles iorec writes, hand-written datafiles with numeric
coordinates, and the output of the Evolver ``dump`` command. Commands after ``read``
are ignored. Other models (string, quadratic, Lagrange, simplex, torus, symmetry
groups) raise `MeshFormatError`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from iorec._evolver_reader import parse_evolver
from iorec._topology import _label_sides, _mesh_arrays, _resolve_mesh_path
from iorec._validation import _check_triangle_mesh
from iorec.rec_appendix import _iorec_version

_SUFFIXES = (".fe", ".dmp")
_FACETS_PER_LINE = 16

Tensions = Union[float, ArrayLike, Mapping[Tuple[int, int], float]]


def load_fe(
    filename: str | Path,
    *,
    exterior_label: int = 0,
    validate: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Read a Surface Evolver datafile or dump as points, triangles and labels.

    Facets with more than three edges are split around a vertex added at their
    centre, as Surface Evolver does. Arguments are those of `load_fe_with_attributes`.

    Returns:
        (points, triangles, labels): float64 (n, 3), int64 (m, 3) and int64 (m, 2).
    """
    points, triangles, labels, _ = load_fe_with_attributes(
        filename,
        exterior_label=exterior_label,
        validate=validate,
    )
    return points, triangles, labels


def load_fe_with_attributes(
    filename: str | Path,
    *,
    exterior_label: int = 0,
    validate: bool = False,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    dict[str, Any],
]:
    """Read a Surface Evolver file with its facet tensions and body volume targets.

    Args:
        filename: a .fe datafile or a .dmp dump.
        exterior_label: label given to facet sides that bound no body.
        validate: also check coordinates, indices and labels as `load_rec` does.

    Returns:
        (points, triangles, labels, attributes): attributes holds `tension` (m,),
        `volume` {label: target}, `vertex_id` (n,) (0 for added centres) and
        `facet_id` (m,), the ids of the Evolver elements.
    """
    path = _resolve_mesh_path(filename, _SUFFIXES)
    text = path.read_text(encoding="latin-1")
    points, triangles, labels, attributes = parse_evolver(text, path, exterior_label)
    if validate:
        _check_triangle_mesh(path, points, triangles, labels)
    return points, triangles, labels, attributes


def save_fe(
    filename: str | Path,
    points: ArrayLike,
    triangles: ArrayLike,
    labels: ArrayLike,
    *,
    exterior_label: int = 0,
    tensions: Tensions | None = None,
    volumes: Mapping[int, float] | str | None = None,
    read_commands: str | None = None,
) -> None:
    """Write a mesh as a Surface Evolver datafile, one body per label but the exterior.

    Args:
        filename: where to write the datafile (.fe).
        points: vertex coordinates, shape (n, 3).
        triangles: vertex indices, shape (m, 3).
        labels: labels of each triangle, shape (m, 2); positive but the exterior.
        exterior_label: the label that gets no body.
        tensions: a number, one per triangle, or {(label_a, label_b): tension}
            covering every interface. Evolver's default is 1.
        volumes: {label: target volume}, or "current" for the loaded volumes.
        read_commands: Evolver commands written after the `read` line.
    """
    pts, tri, lab = _mesh_arrays(points, triangles, labels)
    _check_bodies(lab, exterior_label)
    edges, signed_edges = _edge_table(tri)
    facet_tension = _facet_tensions(tensions, lab)
    body_labels = [int(x) for x in np.unique(lab) if x != exterior_label]
    lines = [
        f"// Surface Evolver datafile written by iorec {_iorec_version()}.",
        f"// Body k is the region of label k; label {exterior_label} is the exterior.",
        "// Facet normals point from labels[:, 0] to labels[:, 1] of the iorec mesh.",
        "keep_originals",
        "",
        "vertices",
        *_rows("%d %r %r %r", np.arange(1, len(pts) + 1), pts),
        "",
        "edges",
        *_rows("%d %d %d", np.arange(1, len(edges) + 1), edges + 1),
        "",
        "faces",
        *_face_rows(signed_edges, facet_tension),
        "",
        "bodies",
        *_body_rows(lab, exterior_label, volumes),
        *_read_section(body_labels, volumes, read_commands),
    ]
    Path(filename).write_text("\n".join(lines) + "\n")


def _check_bodies(labels: NDArray[np.int64], exterior_label: int) -> None:
    bodies = np.unique(labels[labels != exterior_label])
    if bodies.size and bodies[0] < 1:
        error = (
            f"label {bodies[0]} cannot be a Surface Evolver body (body numbers are "
            "positive); make it the exterior label."
        )
        raise ValueError(error)


def _edge_table(
    triangles: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Unique edges (tail < head) and each triangle's signed 1-based edge loop.

    The loop of triangle (v0, v1, v2) is v0->v1, v1->v2, v2->v0, so the vertex
    cycle read back from it starts at v0 and reproduces the triangle.
    """
    half_edges = np.stack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
        axis=1,
    ).reshape(-1, 2)
    if (half_edges[:, 0] == half_edges[:, 1]).any():
        error = "a triangle has a repeated vertex."
        raise ValueError(error)
    low, high = half_edges.min(axis=1), half_edges.max(axis=1)
    base = int(high.max(initial=0)) + 1
    keys, inverse = np.unique(low * base + high, return_inverse=True)
    edges = np.column_stack([keys // base, keys % base])
    sign = np.where(half_edges[:, 0] < half_edges[:, 1], 1, -1)
    signed = (sign * (inverse.reshape(-1) + 1)).reshape(-1, 3)
    return edges, signed


def _facet_tensions(
    tensions: Tensions | None,
    labels: NDArray[np.int64],
) -> NDArray[np.float64] | None:
    if tensions is None:
        return None
    if isinstance(tensions, Mapping):
        by_pair = {tuple(sorted(pair)): float(t) for pair, t in tensions.items()}
        pairs = [tuple(sorted(pair)) for pair in labels.tolist()]
        missing = sorted(set(pairs) - set(by_pair))
        if missing:
            error = f"no tension given for the interfaces {missing}."
            raise ValueError(error)
        return np.asarray([by_pair[pair] for pair in pairs], dtype=np.float64)
    values = np.broadcast_to(np.asarray(tensions, dtype=np.float64), (len(labels),))
    if not np.isfinite(values).all():
        error = "tensions must be finite."
        raise ValueError(error)
    return values


def _rows(pattern: str, ids: NDArray[np.int64], values: NDArray[Any]) -> list[str]:
    """Format `id values...` rows with one `%` operation per row pattern."""
    if not len(ids):
        return []
    table = np.empty((len(ids), 1 + values.shape[1]), dtype=object)
    table[:, 0] = ids.tolist()
    table[:, 1:] = values.tolist()
    text = ((pattern + "\n") * len(ids)) % tuple(table.reshape(-1).tolist())
    return text.rstrip("\n").split("\n")


def _face_rows(
    signed_edges: NDArray[np.int64],
    tension: NDArray[np.float64] | None,
) -> list[str]:
    ids = np.arange(1, len(signed_edges) + 1)
    if tension is None:
        return _rows("%d %d %d %d", ids, signed_edges)
    values = np.empty((len(ids), 4), dtype=object)
    values[:, :3] = signed_edges.tolist()
    values[:, 3] = tension.tolist()
    return _rows("%d %d %d %d tension %r", ids, values)


def _body_rows(
    labels: NDArray[np.int64],
    exterior_label: int,
    volumes: Mapping[int, float] | str | None,
) -> list[str]:
    """One body per label, continued over several lines with a backslash."""
    if isinstance(volumes, str) and volumes != "current":
        error = f"volumes must be a mapping, 'current' or None, got {volumes!r}."
        raise ValueError(error)
    targets = volumes if isinstance(volumes, Mapping) else {}
    sides = _label_sides(labels, exterior_label)
    unknown = sorted(set(targets) - {label for label, _, _ in sides})
    if unknown:
        error = f"volume targets given for labels {unknown}, which are not bodies."
        raise ValueError(error)
    rows = []
    for label, front, back in sides:
        signed = np.concatenate([front + 1, -(back + 1)]).tolist()
        chunks = [
            " ".join(map(str, signed[k : k + _FACETS_PER_LINE]))
            for k in range(0, len(signed), _FACETS_PER_LINE)
        ]
        row = f"{label} " + " \\\n    ".join(chunks)
        if label in targets:
            row += f" volume {float(targets[label])!r}"
        rows.append(row)
    return rows


def _read_section(
    body_labels: list[int],
    volumes: Mapping[int, float] | str | None,
    read_commands: str | None,
) -> list[str]:
    commands = []
    if volumes == "current":
        commands += [f"set body[{k}] target body[{k}].volume" for k in body_labels]
    if read_commands:
        commands.append(read_commands.rstrip("\n"))
    return ["", "read", *commands] if commands else []
