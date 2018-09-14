import evostencils.stencils.constant as constant
import evostencils.stencils.periodic as periodic
import copy
#entries = [
#        ((0,), 2),
#        ((-1,), -1),
#        ((1,), -1)
#]
entries = [
        (( 0, -1), -1.0),
        ((-1,  0), -1.0),
        (( 0,  0),  4.0),
        (( 1,  0), -1.0),
        (( 0,  1), -1.0)
    ]
stencil1 = constant.Stencil(entries)
stencil2 = constant.Stencil(copy.deepcopy(entries))

a = constant.scale(2, stencil1)
b = constant.add(stencil1, stencil2)
c = constant.sub(stencil1, stencil2)
d = constant.mul(stencil1, stencil2)
e = constant.diagonal(stencil1)
f = constant.upper(stencil1)
g = constant.lower(stencil1)
inv_diag = constant.inverse(constant.diagonal(stencil1))
jacobi = constant.mul(constant.inverse(constant.diagonal(stencil1)), constant.add(constant.lower(stencil1), constant.upper(stencil1)))
periodic_stencil = periodic.block_diagonal(stencil1, (2, 2))
print(periodic.count_number_of_entries(periodic_stencil))
tmp = periodic.add(periodic_stencil, constant.get_unit_stencil(2))
print(periodic.count_number_of_entries(tmp))
pass
