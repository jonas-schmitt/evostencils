from evostencils.stencils import constant, periodic

a = constant.Stencil([((0, 0), 4), ((0, 1), -1)], 2)
b = constant.Stencil([((0, 0), 4), ((0, -1), -1)], 2)

c = periodic.Stencil([[a, b]], 2)
nentries = periodic.count_number_of_entries(c)
foo = 1
