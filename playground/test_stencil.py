import evostencils.stencil as stencil
import copy

stencil1 = stencil.Stencil(((( 0,), -2),
            ((-1,),  1),
            (( 1,),  1)))
stencil2 = stencil.Stencil(copy.deepcopy(stencil1.entries))

a = stencil.scale(2, stencil1)
b = stencil.add(stencil1, stencil2)
c = stencil.sub(stencil1, stencil2)
d = stencil.mul(stencil1, stencil2)
e = stencil.diagonal(stencil1)
f = stencil.upper(stencil1)
g = stencil.lower(stencil1)
pass
