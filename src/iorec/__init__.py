"""IO for .rec meshes in 3D or 2D.

Matthieu Perez (2023)
"""

from iorec._validation import MeshFormatError, RecFormatError
from iorec.obj import load_obj, save_obj
from iorec.ply import load_ply, save_ply
from iorec.rec_2d import load_2d_rec, load_2d_rec_with_metadata, save_2d_rec
from iorec.rec_3d import load_rec, load_rec_with_metadata, save_rec
from iorec.rec_appendix import load_rec_appendix, write_rec_appendix
from iorec.surface_evolver import load_fe, load_fe_with_attributes, save_fe
from iorec.threemf import load_3mf, save_3mf

__all__ = (
    "MeshFormatError",
    "RecFormatError",
    "load_2d_rec",
    "load_2d_rec_with_metadata",
    "load_3mf",
    "load_fe",
    "load_fe_with_attributes",
    "load_obj",
    "load_ply",
    "load_rec",
    "load_rec_appendix",
    "load_rec_with_metadata",
    "save_2d_rec",
    "save_3mf",
    "save_fe",
    "save_obj",
    "save_ply",
    "save_rec",
    "write_rec_appendix",
)
