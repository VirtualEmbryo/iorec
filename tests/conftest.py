"""Shared fixtures: test meshes and a comparison of meshes up to ordering."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import iorec

ASSETS = Path(__file__).parent / "assets"
Mesh = tuple[np.ndarray, np.ndarray, np.ndarray]


@pytest.fixture
def cells() -> Mesh:
    """Sixteen closed cells in a medium (label 0), 17000 triangles."""
    return iorec.load_rec(ASSETS / "binary.rec")


@pytest.fixture
def open_cells() -> Mesh:
    """Four labels whose surfaces meet along a free boundary: shells are not closed."""
    return iorec.load_rec(ASSETS / "text.rec")


def canonical(points: np.ndarray, triangles: np.ndarray, labels: np.ndarray) -> list:
    """Sorted triangles as coordinates and labels, whatever the storage order.

    Each triangle is reversed when needed so that its first label is the smaller one
    (the same interface, described the other way round), then its vertex cycle starts
    at its smallest corner.
    """
    rows = []
    for tri, lab in zip(triangles.tolist(), labels.tolist()):
        if lab[0] > lab[1]:
            tri, lab = tri[::-1], lab[::-1]
        corners = [tuple(points[v].tolist()) for v in tri]
        start = corners.index(min(corners))
        rows.append((*corners[start:], *corners[:start], *lab))
    return sorted(rows)


@pytest.fixture
def same_mesh() -> Callable[[Mesh, Mesh], bool]:
    """Whether two meshes have the same labelled triangles, up to ordering."""

    def compare(first: Mesh, second: Mesh) -> bool:
        return canonical(*first) == canonical(*second)

    return compare


def signed_volumes(
    points: np.ndarray,
    triangles: np.ndarray,
    labels: np.ndarray,
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
