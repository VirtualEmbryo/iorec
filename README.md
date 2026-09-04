# iorec

Small utility to load and save .rec mesh files from MultiTracker.

It works with both 3D and 2D .rec meshes.

## Installation

`uv add iorec` if you have configured uv,

or

`embryo-pip install iorec` if you have installed embryo-pip the old way.

## Usage

There's four function: 

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

### load_rec

Load a 3D .rec file into three numpy arrays of points, triangles and multi-material labels.

The file can be in binary or text mode.

```python
import iorec

# with a string :
points, triangles, labels = iorec.load_rec("file.rec")

# or with a pathlib.Path :
from pathlib import Path

points, triangles, labels = iorec.load_rec(Path("file.rec"))

# points: np.array of shape (num_points, 3) (float64)
# triangles: np.array of shape (num_triangles, 3) (ulonglong)
# labels: np.array of shape (num_triangles, 2) (uint)

# If you give a file that does not exist, it raises a FileNotFoundError:
iorec.load_rec("NOT_AN_EXISTING_FILE.rec")  # raises FileNotFoundError
# If you give a file that is not a .rec or a .arec, it raises a ValueError:
iorec.load_rec("tests/assets/README.md")  # raises ValueError

# Be sure have the rights to access the file too, etc...
```

### save_rec

Save three numpy arrays of points, triangles and multi-material labels as a .rec mesh file.

The file can be in binary or text mode. Binary is more compact but might not be universally understood.

Arguments are:
- filename (str | Path): where to save the mesh file.
- points (NDArray[np.float64]): Mesh points.
- triangles (NDArray[np.ulonglong]): Mesh triangles (topology).
- labels (NDArray[np.uint]): Triangles multimaterial labels.
- binary_mode (bool, optional): Choose binary mode. Defaults to False.

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

# or arrays from your own code, of course !
points, segments, labels = iorec.load_rec("tests/assets/text.rec")

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