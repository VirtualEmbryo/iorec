"""IO for .rec meshes in 3D or 2D.

Matthieu Perez (2023)
"""

from iorec.rec_2d import load_2d_rec, save_2d_rec
from iorec.rec_3d import load_rec, save_rec
from iorec.rec_appendix import load_rec_appendix, parse_rec_appendix, write_rec_appendix

__all__ = (
    "load_rec",
    "save_rec",
    "load_2d_rec",
    "save_2d_rec",
    "load_rec_appendix",
    "parse_rec_appendix",
    "write_rec_appendix",
)
