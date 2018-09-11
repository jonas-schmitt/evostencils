import evostencils.stencils.constant as stencils
import copy

stencil1 = stencils.Stencil((((0,), -2),
                             ((-1,),  1),
                             (( 1,),  1)))
stencil2 = stencils.Stencil(copy.deepcopy(stencil1.entries))

a = stencils.scale(2, stencil1)
b = stencils.add(stencil1, stencil2)
c = stencils.sub(stencil1, stencil2)
d = stencils.mul(stencil1, stencil2)
e = stencils.diagonal(stencil1)
f = stencils.upper(stencil1)
g = stencils.lower(stencil1)
inv_diag = stencils.inverse(stencils.diagonal(stencil1))
jacobi = stencils.mul(stencils.inverse(stencils.diagonal(stencil1)), stencils.add(stencils.lower(stencil1), stencils.upper(stencil1)))
pass
