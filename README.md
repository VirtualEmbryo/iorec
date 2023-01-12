# .rec Loader

Small utility to load .rec files from MultiTracker.

## Installation

`pip install -i https://gitlab.college-de-france.fr/virtualembryo/embryo-pip rec_loader`

## Usage

There's only one function: `open_rec`.

```python
import rec_loader

# with a string :
points, triangles, labels = rec_loader.open_rec("file.rec")

# or with a pathlib.Path :
from pathlib import Path

points, triangles, labels = rec_loader.open_rec(Path("file.rec"))

# points: np.array of shape (num_points, 3) (float64)
# triangles: np.array of shape (num_triangles, 3) (uint64)
# labels: np.array of shape (num_triangles, 2) (uint64)

# If you give a file that does not exist, it raises a FileNotFoundError:
rec_loader.open_rec("NOT_AN_EXISTING_FILE.rec")  # raises FileNotFoundError
# If you give a file that is not a .rec or a .arec, it raises a ValueError:
rec_loader.open_rec("tests/assets/README.md")  # raises ValueError

# Be sure have the rights to access the file too, etc...
```
