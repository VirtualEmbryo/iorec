# iorec

Small utility to load and save .rec mesh files from MultiTracker.

It works with both 3D and 2D .rec meshes. 3D meshes can also be read and written as
Surface Evolver datafiles (`.fe`, and `.dmp` dumps), PLY, OBJ (Blender) and 3MF (3D
printing), keeping the two material labels of every triangle (see "Other formats"
below).

## Installation

`uv add iorec` if you have configured uv,

or

`embryo-pip install iorec` if you have installed embryo-pip the old way.

## Usage

Functions:

For 3D .rec meshes:
- iorec.load_rec()
- iorec.save_rec()

For 2D .rec meshes:
- iorec.load_2d_rec()
- iorec.save_2d_rec()

Plus the optional metadata appendix (see below):
- iorec.load_rec_appendix() — shared by both 2D and 3D
- iorec.load_rec_with_metadata() — 3D, combines load_rec() and load_rec_appendix()
- iorec.load_2d_rec_with_metadata() — 2D, combines load_2d_rec() and load_rec_appendix()

Every reader takes an optional `validate=True` and then raises `iorec.RecFormatError` on
a malformed file (see "Validation on read" below).

Other 3D formats (see "Other formats" below):
- iorec.load_fe(), iorec.load_fe_with_attributes(), iorec.save_fe() — Surface Evolver
- iorec.load_ply(), iorec.save_ply()
- iorec.load_obj(), iorec.save_obj()
- iorec.load_3mf(), iorec.save_3mf()
- iorec.load_mesh(), iorec.save_mesh(), iorec.convert() — format chosen by the suffix
- `iorec-convert SOURCE TARGET` (or `python -m iorec SOURCE TARGET`) — conversion from
  the command line

Their readers raise `iorec.MeshFormatError` (of which `RecFormatError` is a subclass)
for a malformed or unsupported file.

### load_rec

Load a 3D .rec file into three numpy arrays of points, triangles and multi-material labels.

The file can be in binary or text mode. Reading a large text file is about 4 times faster
with `pip install iorec[pyarrow]` (installs `pyarrow`): same arrays, bit for bit, and unusual
formatting falls back to the standard reader.

```python
import iorec

# with a string :
points, triangles, labels = iorec.load_rec("file.rec")

# or with a pathlib.Path :
from pathlib import Path

points, triangles, labels = iorec.load_rec(Path("file.rec"))

# points: np.array of shape (num_points, 3) (float64)
# triangles: np.array of shape (num_triangles, 3) (int64)
# labels: np.array of shape (num_triangles, 2) (int64)

# If you give a file that does not exist, it raises a FileNotFoundError:
iorec.load_rec("NOT_AN_EXISTING_FILE.rec")  # raises FileNotFoundError
# If you give a file that is not a .rec or a .arec, it raises a ValueError:
iorec.load_rec("tests/assets/README.md")  # raises ValueError

# Be sure have the rights to access the file too, etc...
```

### Validation on read

Every reader takes an optional `validate=True` keyword. Without it (the default), files
are read exactly as before, without checks: a truncated file or a 2D file read with
`load_rec` can come back as wrong arrays. With it, the reader checks the file and raises
`iorec.RecFormatError` (a `ValueError` subclass) naming the file and the first bad
record:

- binary files: size consistent with the declared counts (a trailing metadata appendix
  is allowed, anything else after the geometry is not);
- text files: valid counts, the declared number of rows, the expected number of columns,
  nothing after the last record except an appendix;
- all files: finite coordinates, vertex indices in `[0, num_points)`, non-negative labels,
  and two different labels on each triangle (segment in 2D).

```python
import iorec

try:
    points, triangles, labels = iorec.load_rec("file.rec", validate=True)
except iorec.RecFormatError as error:
    print(error)
```

### save_rec

Save three numpy arrays of points, triangles and multi-material labels as a .rec mesh file.

The file can be in binary or text mode. Binary is more compact but might not be universally understood.

Arguments are:
- filename (str | Path): where to save the mesh file.
- points (NDArray[np.float64]): Mesh points.
- triangles (NDArray[np.int64]): Mesh triangles (topology).
- labels (NDArray[np.int64]): Triangles multimaterial labels.
- binary_mode (bool, optional): Choose binary mode. Defaults to False.
- metadata (dict, optional, keyword only): fields of a trailing metadata appendix (see
  below). Defaults to None (no appendix).
- written_by (str, optional, keyword only): name and version of the program that
  produced the mesh, stamped in the appendix. Defaults to None (iorec's own version).

Arrays of other dtypes are converted to the file's types. Both modes store every value
exactly: binary mode writes float64 coordinates, int64 indices and int32 labels,
little-endian on every machine; text mode writes each coordinate with the shortest text
that reads back to the same float64 (Python's `repr`), so a text file read back gives
the same bits.

```python
import iorec
import numpy as np

# or arrays from your own code, of course !
points, triangles, labels = iorec.load_rec("tests/assets/text.rec")

# Save in text format
filename = "txt.rec"
iorec.save_rec(filename, points, triangles, labels, binary_mode=False)

# Check: if we load the saved mesh file, 
# we find back the original points, triangles and labels.
saved_points, saved_triangles, saved_labels = iorec.load_rec(filename)

assert np.array_equal(points, saved_points)
assert np.array_equal(triangles, saved_triangles)
assert np.array_equal(labels, saved_labels)
```


### load_2d_rec

Similar to load_rec in 3D. See its documentation.

```python
import iorec
points, segments, labels = iorec.load_2d_rec("tests/assets/text_2d.rec")
```


### save_2d_rec

Similar to save_rec in 3D. See its documentation.

```python
import iorec
import numpy as np

# or arrays from your own code, of course !
points, segments, labels = iorec.load_2d_rec("tests/assets/text_2d.rec")

filename = "txt2d.rec"
iorec.save_2d_rec(filename, points, segments, labels, binary_mode=False)

saved_points, saved_segments, saved_labels = iorec.load_2d_rec(filename)

assert np.array_equal(points, saved_points)
assert np.array_equal(segments, saved_segments)
assert np.array_equal(labels, saved_labels)
```

### `.rec` / `.arec` metadata appendix

`save_rec`/`save_2d_rec` can optionally attach a small metadata appendix to a
`.rec`/`.arec` file: pass a `metadata={...}` dict, and read it back later with
`iorec.load_rec_appendix(filename)`.

The appendix is a trailing block of ASCII `# key = value` lines, written after the
geometry data — never before it, because every reader (this one included, in both text
and binary mode) reads a count and then exactly that many values, so a trailing block is
invisible to it, while a leading one would break parsing outright:

```
# rec-appendix 1
# written_by = iorec <version>
# coordinate_frame = voxel_index
# axis_order = zyx
# spacing = 1.0 0.325 0.325
# spacing_unit = um
# source_shape = 194 200 199
# crop_offset = 0 0 0
```

The fields above are illustrative, not exhaustive — `metadata` accepts any keys.
`spacing`, `source_shape`, and `crop_offset` are parsed back as tuples of `float`/`int`
if present; any other key round-trips as a raw string.

```python
import iorec

iorec.save_rec(
    "mesh.rec",
    points,
    triangles,
    labels,
    metadata={"coordinate_frame": "voxel_index", "spacing": (1.0, 0.325, 0.325)},
)

metadata = iorec.load_rec_appendix("mesh.rec")
# metadata == {"_version": 1, "written_by": "iorec <version>",
#              "coordinate_frame": "voxel_index", "spacing": (1.0, 0.325, 0.325)}
```

**A file with no appendix must be treated as carrying no metadata at all** —
`load_rec_appendix` returns `None` in that case, and every file saved before this
feature existed (or without a `metadata` argument) has no appendix. In particular,
`None` must never be treated as "isotropic/default spacing" or any other default value.

Metadata values must be ASCII — a non-ASCII value raises before anything is written to
disk. The `written_by`/`_version` keys are reserved for the appendix's own bookkeeping:
they are ignored if present in a `metadata` dict you pass in, so a dict returned by
`load_rec_appendix` can be fed straight back into `save_rec`/`save_2d_rec` without
duplicating or corrupting them.

To read the mesh and its metadata in a single call, use `load_rec_with_metadata` (3D) or
`load_2d_rec_with_metadata` (2D) instead of calling `load_rec`/`load_2d_rec` and
`load_rec_appendix` separately:

```python
import iorec

points, triangles, labels, metadata = iorec.load_rec_with_metadata("mesh.rec")
# metadata is None if the file has no appendix
```

## Other formats: Surface Evolver, PLY, OBJ, 3MF

All formats share the convention of `.rec` files: `points` (n, 3), `triangles` (m, 3)
and `labels` (m, 2), where the normal of triangle `i` (right-hand rule on its vertex
order) points **from `labels[i, 0]` to `labels[i, 1]`**. Label 0 is the exterior
medium by default (`exterior_label=0` wherever it matters).

| suffix | format | what is stored | round trip back to `.rec` |
|---|---|---|---|
| `.fe` (read also `.dmp`) | Surface Evolver datafile | one body per label, facet tensions, volume targets | same arrays, bit for bit |
| `.ply` | PLY, binary or text | the nonmanifold mesh, labels as face properties `label0`, `label1` | same arrays, bit for bit |
| `.obj` + `.mtl` | Wavefront OBJ | one coloured object per interface, or per cell | same mesh (*) |
| `.3mf` | 3MF (3D printing) | one closed, outward surface per cell, colour, unit, scale, Cura extruder | same mesh (*) |

(*) Same points and labelled triangles, up to the order of triangles (and of vertices
for 3MF) and the orientation convention: a triangle with labels `(b, a)`, `b > a`,
may come back reversed with labels `(a, b)`. It is the same interface.

```python
import iorec

points, triangles, labels = iorec.load_rec("tests/assets/binary.rec")

iorec.save_fe("cells.fe", points, triangles, labels, volumes="current")
iorec.save_ply("cells.ply", points, triangles, labels)
iorec.save_obj("cells.obj", points, triangles, labels)  # and cells.mtl
iorec.save_3mf("cells.3mf", points, triangles, labels, scale=5.0)

# Or choose the format by suffix, and convert files directly:
points, triangles, labels = iorec.load_mesh("cells.ply")
iorec.convert("cells.fe", "cells_copy.rec")
```

From the command line:

```
iorec-convert cells.rec cells.fe --fix-volumes --tension-file tensions.txt
iorec-convert cells.dmp cells.rec --binary
iorec-convert cells.rec cells.obj --group-by cell
iorec-convert cells.rec cells.3mf --unit micron --scale 500 --extruder 3=1
```

`iorec.convert` keeps the metadata appendix from `.rec` to `.rec`/`.arec`; the other
formats do not hold it, and a warning says so.

### Surface Evolver (`.fe`, `.dmp`)

Surface Evolver describes a multimaterial surface with bodies: body `k` lists its
facets, `+f` when the normal of facet `f` points out of the body, `-f` when it points
in. `save_fe` writes body `k` for label `k` (labels need not be consecutive, and
`keep_originals` makes `body[k]` refer to label `k` in Evolver commands) and no body
for the exterior. Each triangle `(v0, v1, v2)` becomes the edge loop
`v0->v1, v1->v2, v2->v0`, so reading the datafile back gives the same arrays.

```python
iorec.save_fe(
    "cells.fe", points, triangles, labels,
    tensions={(0, 1): 0.5, (1, 2): 1.0, ...},  # or a number, or one value per triangle
    volumes="current",            # or {label: target volume}, or None
    read_commands="g 10",         # Evolver commands run after loading
)
points, triangles, labels, attributes = iorec.load_fe_with_attributes("cells.dmp")
attributes["tension"]  # per triangle, 1 where unset
attributes["volume"]   # {label: target volume}
```

`load_fe` reads the files written by `save_fe`, hand-written datafiles with numeric
coordinates (facets with more than three edges are split around a centre vertex, as
Evolver does), and the output of the Evolver `dump` command. Evolver dumps write
coordinates with 15 significant digits, so points read from a dump differ from the
float64 originals by a few 1e-15 (relative). Commands after `read` are ignored.
Other models (string, quadratic, Lagrange, simplex, torus, symmetry groups),
parametric boundaries and coordinates given by expressions raise
`iorec.MeshFormatError`: load such a file in Evolver and read its dump instead.
Bodies closed partly by constraints (walls without facets) are read, but their volume
cannot be computed from the facets alone.

### PLY (`.ply`)

The exact exchange format: double coordinates, int vertex indices and two int face
properties `label0`, `label1`, in binary little-endian (default) or text
(`binary_mode=False`); `colors=True` adds a colour per face. Any PLY with `vertex`
and `face` elements can be read; faces without labels get labels `(1, 0)`. Blender
imports the geometry of a PLY but not its face properties: use OBJ for Blender.

### OBJ (`.obj` + `.mtl`), for Blender

`save_obj` writes one OBJ object per interface (`interface_<a>_<b>`, a < b, normals
from a to b), or with `group_by="cell"` one closed surface per cell (`cell_<k>`,
normals outward; a triangle between two cells is then written in both). Each object
gets a material of the same name and colour in the `.mtl` file. In Blender, *File >
Import > Wavefront (.obj)* gives one coloured object per interface or per cell. Files
re-exported by Blender hold a copy of shared vertices in each object: read them with
`load_obj(..., merge_vertices=True)`.

### 3MF (`.3mf`), for 3D printing

3MF holds closed surfaces, so `save_3mf` writes the surface of each cell (all labels
but the exterior) as one part of an assembly; cells stay in place when the slicer
moves the assembly. Coordinates are written exactly; the printed size is
`coordinates * scale` in `unit` (`"micron"`, `"millimeter"`, ...), and the lowest
corner of the assembly is put on the build plate. A cell whose surface is not closed
is still written, with a warning, since a slicer may repair or reject it.

For an Ultimaker S5 (two extruders) with Cura, `extruders={label: 0 or 1}` sets the
extruder of each cell (Cura's `extruder_nr`, 0 for the first one), for instance to
print one cell in a second colour or in a soluble material:

```python
# An aggregate about 6 units wide, printed 10 cm wide; cell 7 on the second extruder.
extruders = {label: 0 for label in range(1, 17)}
extruders[7] = 1
iorec.save_3mf("aggregate.3mf", points, triangles, labels, unit="millimeter",
               scale=100 / 6, extruders=extruders)
```

Open the file in Cura (*File > Open*): Cura reads the extruder of each part from
`cura:extruder_nr` and keeps the parts together as a group. Other slicers
(PrusaSlicer, Bambu Studio) read the parts and their colours; their extruders are set
in the slicer.

## License

Copyright (C) 2023-2026 Matthieu Perez and the iorec contributors.

iorec is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version
3 of the License, or (at your option) any later version (SPDX: `GPL-3.0-or-later`).

iorec is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License in [LICENSE](LICENSE) for more details.
