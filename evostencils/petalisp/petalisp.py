import cl4py
import math
from expressions import base,multigrid,transformations
from evostencils import stencils

def apply_stencil(stencil, grid):
    # Stencil is an (offset value) list.  Each offset is a list.
    dim = len(stencil[0][0])
    pass
