# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2023-2026 the iorec authors
"""Parser of Surface Evolver datafiles and dumps into labelled triangles.

The text is cleaned of comments, then cut into its header and element sections. A
section whose lines all have the plain layout of iorec datafiles and Evolver dumps
(`id` then numbers, ids consecutive) is converted in one numpy call; any other is read
line by line. Elements refer to each other by id, resolved with array lookups.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from iorec._validation import MeshFormatError

_SECTIONS = {
    "vertices": "vertices",
    "edges": "edges",
    "faces": "faces",
    "facets": "faces",
    "bodies": "bodies",
}
_UNSUPPORTED = {
    "string": "the string model",
    "quadratic": "the quadratic model",
    "lagrange": "the Lagrange model",
    "simplex_representation": "the simplex model",
    "torus": "the torus model",
    "torus_filled": "the torus model",
    "symmetry_group": "symmetry groups",
    "#include": "include files",
}
_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?")
_ONE_LINE_BLOCK = re.compile(r"/\*[^\n]*?\*/")
_COMMENTS = re.compile(r"//[^\n]*|/\*.*?(?:\*/|\Z)", re.DOTALL)
_KEYWORD_LINE = re.compile(
    r"^[ \t]*(vertices|edges|faces|facets|bodies|read)[ \t]*\r?$",
    re.MULTILINE | re.IGNORECASE,
)
_WHITESPACE = str.maketrans(",;:", "   ")

# (line number, tokens) of each element line.
_Rows = list[tuple[int, list[str]]]
# Text of a section and the number of its first line.
_Section = tuple[str, int]
_Mesh = tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], dict[str, Any]]
_Loops = NDArray[np.int64] | list[list[str]]


def parse_evolver(text: str, source: Path, exterior_label: int) -> _Mesh:
    """Points, triangles, labels and attributes of a datafile or dump text."""
    sections = _sections(_clean(text), source)
    vertex_ids, coordinates = _vertex_block(sections["vertices"], source)
    edge_ids, edge_ends = _edge_block(sections["edges"], source)
    face_ids, loops, tensions = _face_block(sections["faces"], source)
    bodies = [
        _body(tokens, f"{source}:{n}")
        for n, tokens in _rows(*sections["bodies"], source)
    ]
    vertex_of = _Lookup(vertex_ids, "vertex", source)
    tails, tail_ok = vertex_of(edge_ends[:, 0])
    heads, head_ok = vertex_of(edge_ends[:, 1])
    if not (tail_ok & head_ok).all():
        bad = edge_ids[~(tail_ok & head_ok)][0]
        error = f"{source}: edge {bad} refers to an undefined vertex."
        raise MeshFormatError(error)
    edges = (_Lookup(edge_ids, "edge", source), tails, heads)
    points, triangles, per_face, centres = _triangulate(
        coordinates,
        face_ids,
        loops,
        edges,
        source,
    )
    front, back = _face_sides(face_ids, bodies, exterior_label, source)
    labels = np.column_stack([front[per_face], back[per_face]])
    labels[labels == -1] = exterior_label
    attributes = {
        "tension": np.where(np.isnan(tensions), 1.0, tensions)[per_face],
        "volume": {label: v for label, _, v in bodies if v is not None},
        "vertex_id": np.concatenate([vertex_ids, np.zeros(centres, np.int64)]),
        "facet_id": face_ids[per_face],
    }
    return points, triangles, labels, attributes


def _clean(text: str) -> str:
    """Drop comments, keep line breaks; commas, colons, semicolons become spaces."""

    def blank(match: re.Match[str]) -> str:
        found = match.group()
        return "" if found.startswith("//") else " " + "\n" * found.count("\n")

    return _COMMENTS.sub(blank, _ONE_LINE_BLOCK.sub(" ", text)).translate(_WHITESPACE)


def _sections(clean: str, source: Path) -> dict[str, _Section]:
    """Text of each element section, after checking the header; stops at `read`."""
    sections = dict.fromkeys(set(_SECTIONS.values()), ("", 1))
    depth, position, previous = 0, 0, None
    for match in _KEYWORD_LINE.finditer(clean):
        depth += clean.count("{", position, match.start())
        depth -= clean.count("}", position, match.start())
        position = match.start()
        if depth > 0:
            continue
        if previous is None:
            _check_header_text(clean[: match.start()], source)
        else:
            name, start = previous
            line = clean.count("\n", 0, start) + 1
            text = sections[name][0] + clean[start : match.start()]
            sections[name] = (
                text,
                line if not sections[name][0] else sections[name][1],
            )
        keyword = match.group(1).lower()
        if keyword == "read":
            return sections
        previous = (_SECTIONS[keyword], match.end())
    if previous is None:
        _check_header_text(clean, source)
    else:
        name, start = previous
        line = clean.count("\n", 0, start) + 1
        sections[name] = (sections[name][0] + clean[start:], line)
    return sections


def _check_header_text(header: str, source: Path) -> None:
    """Check the top-level lines of the header, skipping procedure bodies in braces."""
    depth = 0
    for number, tokens in _logical_lines(header):
        if depth == 0:
            _check_header(tokens, f"{source}:{number}")
        depth += sum(t.count("{") - t.count("}") for t in tokens)


def _logical_lines(text: str, first_line: int = 1) -> Iterator[tuple[int, list[str]]]:
    """Yield (line number, tokens) of each non-empty line of cleaned text.

    A line ending with a backslash is joined to the next one.
    """
    pending: list[str] = []
    start = first_line
    for number, line in enumerate(text.split("\n"), start=first_line):
        splice = "\\" in line and line.rstrip().endswith("\\")
        if not pending:
            start = number
        pending += (line.rstrip()[:-1] if splice else line).split()
        if splice or not pending:
            continue
        yield start, pending
        pending = []
    if pending:
        yield start, pending


def _rows(text: str, first_line: int, source: Path) -> _Rows:
    """Element lines of a section; each must start with an element number."""
    rows = list(_logical_lines(text, first_line))
    for number, tokens in rows:
        if not tokens[0][0].isdigit():
            error = (
                f"{source}:{number}: unexpected {tokens[0]!r} in an element section."
            )
            raise MeshFormatError(error)
    return rows


def _table(
    text: str,
    columns: int,
    dtype: type[np.float64] | type[np.int64],
) -> NDArray[Any] | None:
    """The section as an array of `columns` numbers per line, or None.

    None unless every token is a number and the first column holds consecutive
    element numbers, which a line with more or fewer values would break.
    """
    if "\\" in text:
        return None
    tokens = text.split()
    if not tokens or len(tokens) % columns:
        return None
    try:
        table = np.array(tokens, dtype=dtype).reshape(-1, columns)
    except ValueError:
        return None
    ids = table[:, 0]
    first = ids[0]
    if not (first >= 1 and (ids == np.arange(first, first + len(ids))).all()):
        return None
    return table


def _check_header(tokens: list[str], where: str) -> None:
    """Raise if a top-level header line selects a model iorec does not read."""
    keyword = tokens[0].lower()
    if keyword in _UNSUPPORTED:
        error = (
            f"{where}: unsupported datafile ({_UNSUPPORTED[keyword]}); iorec reads "
            "3D soapfilm meshes in the linear model."
        )
        raise MeshFormatError(error)
    expected = {"space_dimension": "3", "surface_dimension": "2"}
    if keyword in expected and tokens[1:2] != [expected[keyword]]:
        value = " ".join(tokens[1:2])
        error = f"{where}: {keyword} {value} is not supported (3D only)."
        raise MeshFormatError(error)


def _is_integer(token: str) -> bool:
    return token.isdigit() or (token[0] in "+-" and token[1:].isdigit())


def _number(token: str) -> float:
    return float(token.replace("d", "e").replace("D", "e"))


def _attribute_number(
    tokens: list[str],
    names: tuple[str, ...],
    where: str,
) -> float | None:
    """Value of the first attribute among `names`, or None if absent."""
    for k, token in enumerate(tokens):
        if token.lower() in names:
            value = tokens[k + 1] if k + 1 < len(tokens) else ""
            if not _NUMBER.fullmatch(value):
                error = f"{where}: {token} must be a number, got {value!r}."
                raise MeshFormatError(error)
            return _number(value)
    return None


def _vertex_block(
    section: _Section,
    source: Path,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Vertex ids and coordinates."""
    table = _table(section[0], 4, np.float64)
    if table is not None and np.isfinite(table).all():
        return table[:, 0].astype(np.int64), table[:, 1:]
    ids, coordinates = [], []
    for number, tokens in _rows(*section, source):
        where = f"{source}:{number}"
        if "boundary" in (t.lower() for t in tokens[1:]):
            error = f"{where}: vertex {tokens[0]} is on a parametric boundary."
            raise MeshFormatError(error)
        values, extra = tokens[1:4], tokens[4:5]
        numeric = len(values) == 3 and all(_NUMBER.fullmatch(t) for t in values)
        if (
            not tokens[0].isdigit()
            or not numeric
            or (extra and _NUMBER.fullmatch(extra[0]))
        ):
            error = (
                f"{where}: vertex {tokens[0]} needs three numeric coordinates, got "
                f"{' '.join(tokens[1:5])!r}; for coordinates given by expressions, "
                "load the file in Surface Evolver and read its dump instead."
            )
            raise MeshFormatError(error)
        ids.append(int(tokens[0]))
        coordinates.append([_number(t) for t in values])
    return np.asarray(ids, np.int64), np.asarray(coordinates, np.float64).reshape(-1, 3)


def _edge_block(
    section: _Section,
    source: Path,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Edge ids and (tail, head) vertex ids, from the first three tokens of a line."""
    table = _table(section[0], 3, np.int64)
    if table is None:
        rows = _rows(*section, source)
        for number, tokens in rows:
            if len(tokens) < 3 or not all(
                t.isdigit() and int(t) > 0 for t in tokens[:3]
            ):
                error = (
                    f"{source}:{number}: an edge needs a number and two vertex numbers."
                )
                raise MeshFormatError(error)
        table = np.array([t[:3] for _, t in rows], dtype=np.int64).reshape(-1, 3)
    return table[:, 0], table[:, 1:]


def _face_block(
    section: _Section,
    source: Path,
) -> tuple[NDArray[np.int64], _Loops, NDArray[np.float64]]:
    """Facet ids, signed edge loops and tensions (NaN where unset).

    The loops are an (m, 3) array when every facet is a plain triangle, and lists
    of tokens otherwise.
    """
    table = _table(section[0], 4, np.int64)
    if table is not None:
        return table[:, 0], table[:, 1:], np.full(len(table), np.nan)
    if "tension" in section[0]:
        table = _table(section[0].replace("tension", " "), 5, np.float64)
        if table is not None and (table[:, 1:4] == np.round(table[:, 1:4])).all():
            triangles = table[:, 1:4].astype(np.int64)
            return table[:, 0].astype(np.int64), triangles, table[:, 4]
    loops: list[list[str]] = []
    tensions, ids = [], []
    for number, tokens in _rows(*section, source):
        end = 1
        while end < len(tokens) and _is_integer(tokens[end]):
            end += 1
        loops.append(tokens[1:end])
        names = ("tension", "density")
        tension = _attribute_number(tokens[end:], names, f"{source}:{number}")
        tensions.append(np.nan if tension is None else tension)
        ids.append(tokens[0])
    return _positive_ids(ids, "facet", source), loops, np.asarray(tensions, np.float64)


def _positive_ids(tokens: list[str], what: str, source: Path) -> NDArray[np.int64]:
    bad = next((t for t in tokens if not t.isdigit() or int(t) == 0), None)
    if bad is not None:
        error = f"{source}: invalid {what} number {bad!r}."
        raise MeshFormatError(error)
    return np.asarray([int(t) for t in tokens], dtype=np.int64)


def _body(tokens: list[str], where: str) -> tuple[int, NDArray[np.int64], float | None]:
    """(label, signed facet ids, volume target) of one body line."""
    end = 1
    while end < len(tokens) and _is_integer(tokens[end]):
        end += 1
    if not tokens[0].isdigit() or int(tokens[0]) == 0:
        error = f"{where}: invalid body number {tokens[0]!r}."
        raise MeshFormatError(error)
    volume = _attribute_number(tokens[end:], ("volume",), where)
    original = _attribute_number(tokens[end:], ("original",), where)
    label = int(tokens[0]) if original is None else int(original)
    facets = np.asarray([int(t) for t in tokens[1:end]], dtype=np.int64)
    return label, facets, volume


class _Lookup:
    """Map element ids to their position in file order."""

    def __init__(self, ids: NDArray[np.int64], what: str, source: Path) -> None:
        self.order = np.argsort(ids, kind="stable")
        self.sorted = ids[self.order]
        repeated = np.flatnonzero(self.sorted[1:] == self.sorted[:-1])
        if repeated.size:
            error = f"{source}: {what} {self.sorted[repeated[0]]} is defined twice."
            raise MeshFormatError(error)

    def __call__(
        self,
        ids: NDArray[np.int64],
    ) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
        """Positions of `ids`, and whether each id exists."""
        if not len(self.sorted):
            return np.zeros(len(ids), np.intp), np.zeros(len(ids), bool)
        found = np.searchsorted(self.sorted, ids).clip(0, len(self.sorted) - 1)
        exists = self.sorted[found] == ids
        return self.order[found], exists


def _triangulate(
    coordinates: NDArray[np.float64],
    face_ids: NDArray[np.int64],
    loops: _Loops,
    edges: tuple[_Lookup, NDArray[np.intp], NDArray[np.intp]],
    source: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.intp], int]:
    """Triangles of all facets in file order, splitting polygons around a centre.

    Returns the points (with added centres), the triangles, the facet of each
    triangle and the number of added centres.
    """
    if isinstance(loops, np.ndarray):
        triangles = _cycles(loops, face_ids, edges, source)
        return coordinates, triangles, np.arange(len(loops), dtype=np.intp), 0
    lengths = np.asarray([len(loop) for loop in loops], dtype=np.intp)
    counts = np.where(lengths == 3, 1, lengths)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.intp)
    triangles = np.empty((int(counts.sum()), 3), dtype=np.int64)
    simple = np.flatnonzero(lengths == 3)
    signed = np.asarray([loops[k] for k in simple], dtype=np.int64).reshape(-1, 3)
    triangles[first[simple]] = _cycles(signed, face_ids[simple], edges, source)
    centres: list[NDArray[np.float64]] = []
    for k in np.flatnonzero(lengths != 3).tolist():
        if lengths[k] < 3:
            error = f"{source}: facet {face_ids[k]} has fewer than three edges."
            raise MeshFormatError(error)
        loop = np.asarray(loops[k], dtype=np.int64).reshape(1, -1)
        cycle = _cycles(loop, face_ids[k : k + 1], edges, source)[0]
        centre = len(coordinates) + len(centres)
        centres.append(coordinates[cycle].mean(axis=0))
        pieces = np.column_stack(
            [np.roll(cycle, 1), cycle, np.full(len(cycle), centre)],
        )
        triangles[first[k] : first[k] + lengths[k]] = pieces
    points = np.vstack([coordinates, *centres]) if centres else coordinates
    per_face = np.repeat(np.arange(len(loops), dtype=np.intp), counts)
    return points, triangles, per_face, len(centres)


def _cycles(
    signed: NDArray[np.int64],
    face_ids: NDArray[np.int64],
    edges: tuple[_Lookup, NDArray[np.intp], NDArray[np.intp]],
    source: Path,
) -> NDArray[np.int64]:
    """Vertex cycles of facets given as closed loops of signed edge ids, one per row."""
    edge_of, tails, heads = edges
    position, exists = edge_of(np.abs(signed).reshape(-1))
    position, exists = position.reshape(signed.shape), exists.reshape(signed.shape)
    forward = signed > 0
    start = np.where(forward, tails[position], heads[position])
    end = np.where(forward, heads[position], tails[position])
    closed = (end == np.roll(start, -1, axis=1)).all(axis=1)
    bad = np.flatnonzero(~exists.all(axis=1) | ~closed)
    if bad.size:
        k = bad[0]
        reason = (
            "an undefined edge" if not exists[k].all() else "edges that do not close"
        )
        error = f"{source}: facet {face_ids[k]} has {reason}."
        raise MeshFormatError(error)
    return start.astype(np.int64)


def _face_sides(
    face_ids: NDArray[np.int64],
    bodies: list[tuple[int, NDArray[np.int64], float | None]],
    exterior_label: int,
    source: Path,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Label of the body each facet points out of (front) and into (back), or -1."""
    front = np.full(len(face_ids), -1, dtype=np.int64)
    back = np.full(len(face_ids), -1, dtype=np.int64)
    if not bodies:
        return front, back
    for label, _, _ in bodies:
        if label == exterior_label:
            error = f"{source}: body {label} has the exterior label {exterior_label}."
            raise MeshFormatError(error)
    signed = np.concatenate([facets for _, facets, _ in bodies])
    owner = np.concatenate([np.full(len(f), label) for label, f, _ in bodies])
    position, exists = _Lookup(face_ids, "facet", source)(np.abs(signed))
    if not exists.all():
        k = np.flatnonzero(~exists)[0]
        error = f"{source}: body {owner[k]} refers to an undefined facet {signed[k]}."
        raise MeshFormatError(error)
    for side, mask in ((front, signed > 0), (back, signed < 0)):
        on_side = position[mask]
        unique, counts = np.unique(on_side, return_counts=True)
        if (counts > 1).any():
            facet = face_ids[unique[counts > 1][0]]
            error = f"{source}: facet {facet} has two bodies on one side."
            raise MeshFormatError(error)
        side[on_side] = owner[mask]
    return front, back
