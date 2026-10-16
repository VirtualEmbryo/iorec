"""Shared mesh helpers for the exchange formats: input checks, cell shells, merging.

A multimaterial mesh is stored once, nonmanifold: each triangle separates the label
`labels[i, 0]`, which its normal (right-hand rule on the vertex order) points away
from, from the label `labels[i, 1]`. Formats that hold one closed surface per material
(3MF, OBJ by cell) store instead the boundary of each label, oriented outward:
`_cell_shells` builds these shells and `_merge_shells` rebuilds the labelled mesh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from iorec._validation import MeshFormatError


def _resolve_mesh_path(filename: str | Path, suffixes: tuple[str, ...]) -> Path:
    """Return the absolute path of an existing file with one of `suffixes`."""
    path = Path(filename).resolve()
    if not path.exists():
        error = f"{path} not found."
        raise FileNotFoundError(error)
    if path.suffix.lower() not in suffixes:
        error = f"{path} is not a {' or '.join(suffixes)} file."
        raise ValueError(error)
    return path


def _mesh_arrays(
    points: ArrayLike,
    triangles: ArrayLike,
    labels: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Convert a mesh to float64/int64 arrays and check shapes, indices and values.

    Raises:
        ValueError: if a shape is wrong, a coordinate is not finite, a vertex index is
            out of range or a label is negative.
    """
    pts = np.asarray(points, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64)
    lab = np.asarray(labels, dtype=np.int64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        error = f"points must have shape (n, 3), got {pts.shape}."
        raise ValueError(error)
    if tri.ndim != 2 or tri.shape[1] != 3:
        error = f"triangles must have shape (m, 3), got {tri.shape}."
        raise ValueError(error)
    if lab.shape != (len(tri), 2):
        error = f"labels must have shape ({len(tri)}, 2), got {lab.shape}."
        raise ValueError(error)
    if not np.isfinite(pts).all():
        error = "points contain a non-finite coordinate."
        raise ValueError(error)
    if tri.size and (tri.min() < 0 or tri.max() >= len(pts)):
        error = f"triangles refer to vertices outside [0, {len(pts)})."
        raise ValueError(error)
    if lab.size and lab.min() < 0:
        error = "labels must be non-negative."
        raise ValueError(error)
    return pts, tri, lab


def _label_sides(
    labels: NDArray[np.int64],
    exterior_label: int | None,
) -> list[tuple[int, NDArray[np.intp], NDArray[np.intp]]]:
    """For each label but the exterior, the triangles where it is first or second label.

    Returns (label, front, back) in increasing label order: `front` are the indices of
    the triangles with `labels[i, 0] == label`, `back` those with `labels[i, 1] ==
    label`, each in increasing order. One sort serves all labels.
    """
    flat = labels.T.reshape(-1)
    order = np.argsort(flat, kind="stable")
    values, starts = np.unique(flat[order], return_index=True)
    ends = np.append(starts[1:], len(flat))
    sides = []
    for label, start, end in zip(values.tolist(), starts, ends):
        if exterior_label is not None and label == exterior_label:
            continue
        indices = order[start:end]
        split = np.searchsorted(indices, len(labels))
        sides.append((label, indices[:split], indices[split:] - len(labels)))
    return sides


def _cell_shells(
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
    exterior_label: int | None,
) -> list[tuple[int, NDArray[np.int64]]]:
    """Return, for each label but the exterior, its boundary triangles oriented outward.

    A triangle is outward for `labels[i, 0]` as stored, and is reversed for
    `labels[i, 1]`. Labels are returned in increasing order.
    """
    return [
        (label, np.concatenate([triangles[front], triangles[back][:, ::-1]]))
        for label, front, back in _label_sides(labels, exterior_label)
    ]


def _open_half_edges(oriented: NDArray[np.int64]) -> int:
    """Count the half-edges of a triangle set that have no opposite half-edge.

    Zero means every edge is shared by triangles of opposite orientation, as in a
    closed, consistently oriented surface.
    """
    half_edges = np.concatenate(
        [oriented[:, [0, 1]], oriented[:, [1, 2]], oriented[:, [2, 0]]],
    )
    if not len(half_edges):
        return 0
    base = int(half_edges.max()) + 1
    forward = np.unique(half_edges[:, 0] * base + half_edges[:, 1])
    backward = half_edges[:, 1] * base + half_edges[:, 0]
    return int(np.count_nonzero(~np.isin(forward, backward)))


def _merge_shells(
    shells: list[tuple[int, NDArray[np.int64]]],
    exterior_label: int,
    source: Path | str,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Rebuild the labelled nonmanifold triangles from outward-oriented cell shells.

    A triangle found in two shells, with opposite orientations, becomes one triangle
    between the two labels; a triangle found in one shell faces the exterior label.
    Vertex indices must already be shared between shells.

    Raises:
        MeshFormatError: if a triangle is in more than two shells, twice in one shell,
            or with the same orientation in two shells.
    """
    if not shells:
        return np.empty((0, 3), np.int64), np.empty((0, 2), np.int64)
    oriented = np.concatenate([tri for _, tri in shells]).astype(np.int64)
    owner = np.concatenate([np.full(len(tri), lab) for lab, tri in shells])
    _, first, inverse, counts = np.unique(
        np.sort(oriented, axis=1),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    inverse = inverse.reshape(-1)
    if (counts > 2).any():
        error = f"{source}: a triangle belongs to more than two cell surfaces."
        raise MeshFormatError(error)
    keep = np.sort(first)
    triangles = oriented[keep]
    labels = np.column_stack([owner[keep], np.full(len(keep), exterior_label)])
    shared = np.flatnonzero(counts[inverse] == 2)
    if shared.size:
        partner = _pair_partners(inverse, shared)
        is_kept = np.isin(shared, keep)
        kept, other = shared[is_kept], partner[is_kept]
        _check_opposite(
            oriented[kept],
            oriented[other],
            owner[kept],
            owner[other],
            source,
        )
        position = np.searchsorted(keep, kept)
        labels[position, 1] = owner[other]
    return triangles, labels


def _pair_partners(
    inverse: NDArray[np.intp],
    shared: NDArray[np.intp],
) -> NDArray[np.intp]:
    """For each occurrence in `shared`, the other occurrence of the same triangle."""
    order = shared[np.argsort(inverse[shared], kind="stable")]
    pairs = order.reshape(-1, 2)
    partner = np.empty(len(inverse), dtype=np.intp)
    partner[pairs[:, 0]] = pairs[:, 1]
    partner[pairs[:, 1]] = pairs[:, 0]
    return partner[shared]


def _check_opposite(
    first: NDArray[np.int64],
    second: NDArray[np.int64],
    first_owner: NDArray[Any],
    second_owner: NDArray[Any],
    source: Path | str,
) -> None:
    """Raise unless each pair of copies has two owners and opposite orientations."""
    same_owner = np.flatnonzero(first_owner == second_owner)
    if same_owner.size:
        error = (
            f"{source}: the surface of label {first_owner[same_owner[0]]} contains "
            "the same triangle twice."
        )
        raise MeshFormatError(error)
    same_orientation = np.flatnonzero(_parity(first) == _parity(second))
    if same_orientation.size:
        k = same_orientation[0]
        error = (
            f"{source}: labels {first_owner[k]} and {second_owner[k]} share a triangle "
            "with the same orientation; a cell surface is not oriented outward."
        )
        raise MeshFormatError(error)


def _parity(triangles: NDArray[np.int64]) -> NDArray[np.bool_]:
    """True for triangles whose vertex cycle is an even permutation of sorted order."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return ((a < b) & (b < c)) | ((b < c) & (c < a)) | ((c < a) & (a < b))


def _merge_coincident_vertices(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Merge vertices with identical coordinates, keeping the order of first use."""
    if not len(points):
        return points, triangles
    _, first, inverse = np.unique(
        points,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return points[first[order]], rank[inverse.reshape(-1)][triangles]


def _fan(polygon: list[int]) -> list[list[int]]:
    """Split a polygon `[v0, ..., vn]` into triangles around `v0`, same orientation."""
    return [
        [polygon[0], polygon[k], polygon[k + 1]] for k in range(1, len(polygon) - 1)
    ]


def _signed_volumes(
    points: NDArray[np.float64],
    triangles: NDArray[np.int64],
    labels: NDArray[np.int64],
) -> dict[int, float]:
    """Volume enclosed by each label, from the divergence theorem on its triangles.

    Exact for a closed cell surface; for an open one, the value depends on the origin.
    """
    p0, p1, p2 = (points[triangles[:, k]] for k in range(3))
    six_volume = np.einsum("ij,ij->i", p0, np.cross(p1, p2))
    volumes = {}
    for label in np.unique(labels):
        outward = six_volume[labels[:, 0] == label].sum()
        inward = six_volume[labels[:, 1] == label].sum()
        volumes[int(label)] = float(outward - inward) / 6.0
    return volumes
