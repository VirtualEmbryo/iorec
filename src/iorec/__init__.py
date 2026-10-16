"""IO for .rec meshes in 3D or 2D.

Matthieu Perez (2023)
"""

from iorec._validation import MeshFormatError, RecFormatError
from iorec.rec_2d import load_2d_rec, load_2d_rec_with_metadata, save_2d_rec
from iorec.rec_3d import load_rec, load_rec_with_metadata, save_rec
from iorec.rec_appendix import load_rec_appendix, write_rec_appendix
from iorec.surface_evolver import load_fe, load_fe_with_attributes, save_fe

__all__ = (
    "MeshFormatError",
    "RecFormatError",
    "load_2d_rec",
    "load_2d_rec_with_metadata",
    "load_fe",
    "load_fe_with_attributes",
    "load_rec",
    "load_rec_appendix",
    "load_rec_with_metadata",
    "save_2d_rec",
    "save_fe",
    "save_rec",
    "write_rec_appendix",
)
