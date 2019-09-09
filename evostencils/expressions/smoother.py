from evostencils.expressions import base, system, partitioning as part
from evostencils.stencils import constant, periodic


def generate_smoother(smoothing_operator, system_operator, approximation, rhs,
                      partitioning=part.Single, relaxation_factor=1):
    residual = base.Residual(system_operator, approximation, rhs)
    correction = base.Multiplication(base.Inverse(smoothing_operator), residual)
    return base.Cycle(approximation, rhs, correction, partitioning, relaxation_factor)


def decoupled_jacobi(operator: system.Operator, approximation, rhs,
                     partitioning=part.Single, relaxation_factor=1):
    smoothing_operator = system.Diagonal(operator)
    return generate_smoother(smoothing_operator, operator, approximation, rhs, partitioning, relaxation_factor)


def collective_jacobi(operator: system.Operator, approximation, rhs,
                      partitioning=part.Single, relaxation_factor=1):
    smoothing_operator = system.ElementwiseDiagonal(operator)
    return generate_smoother(smoothing_operator, operator, approximation, rhs, partitioning, relaxation_factor)


def collective_block_jacobi(operator: system.Operator, approximation, rhs,
                            block_size, partitioning=part.Single, relaxation_factor=1):
    entries = []
    for row in operator.entries:
        entries.append([])
        for entry in row:
            stencil = entry.generate_stencil()
            block_diagonal = periodic.block_diagonal(stencil, block_size)
            new_entry = base.Operator(f'{entry.name}_bd', entry.grid, base.ConstantStencilGenerator(block_diagonal))
            entries[-1].append(new_entry)
    smoothing_operator = system.Operator(f'{operator.name}_bd', entries)
    return generate_smoother(smoothing_operator, operator, approximation, rhs, partitioning, relaxation_factor)


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
