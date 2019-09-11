from evostencils.expressions import base, system, partitioning as part
from evostencils.stencils import constant, periodic


def generate_decoupled_jacobi(operator: system.Operator):
    return system.Diagonal(operator)


def generate_collective_jacobi(operator: system.Operator):
    return system.ElementwiseDiagonal(operator)


def generate_collective_block_jacobi(operator: system.Operator, block_size):
    entries = []
    for i, row in enumerate(operator.entries):
        entries.append([])
        for j, entry in enumerate(row):
            stencil = entry.generate_stencil()
            block_diagonal = periodic.block_diagonal(stencil, block_size)
            new_entry = base.Operator(f'{operator.name}_{i}{j}_block_diag', entry.grid, base.ConstantStencilGenerator(block_diagonal))
            entries[-1].append(new_entry)
    return system.Operator(f'{operator.name}_block_diag', entries)


# TODO adapt generation of solve locally to resolve dependencies accurately
"""
def decoupled_block_jacobi(operator: system.Operator, approximation, rhs,
                            block_size, partitioning=part.Single, relaxation_factor=1):
    entries = []
    for i, row in enumerate(operator.entries):
        entries.append([])
        for j, entry in enumerate(row):
            if i == j:
                stencil = entry.generate_stencil()
                block_diagonal = periodic.block_diagonal(stencil, block_size)
                new_entry = base.Operator(f'{entry.name}_bd', entry.grid, base.ConstantStencilGenerator(block_diagonal))
            else:
                new_entry = base.ZeroOperator(entry.grid)
            entries[-1].append(new_entry)
    smoothing_operator = system.Operator(f'{operator.name}_bd', entries)
    return generate_smoother(smoothing_operator, operator, approximation, rhs, partitioning, relaxation_factor)
"""
