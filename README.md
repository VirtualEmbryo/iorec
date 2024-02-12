# iorec

Small utility to load and save .rec mesh files from MultiTracker.

## Installation

`embryo-pip install iorec`

## Usage

There's two function: 

### load_rec

Load a .rec file into three numpy arrays of points, triangles and multi-material labels.

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
