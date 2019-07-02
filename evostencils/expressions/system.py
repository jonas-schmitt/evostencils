from evostencils.expressions import base, multigrid
import functools


class Operator(base.Entity):

    def __init__(self, name, entries):
        self._name = name
        self._entries = entries
        shape = [0, 0]
        for row in entries:
            entry = row[0]
            shape[0] += entry.shape[0]
        for entry in entries[0]:
            shape[1] += entry.shape[1]
        self._shape = tuple(shape)
        super().__init__()

    @property
    def entries(self):
        return self._entries

    @property
    def grid(self):
        return list(map(lambda entry: entry.grid, self.entries[0]))


class ZeroOperator(Operator):
    def __init__(self, grid: [base.Grid], name='0'):
        entries = [[base.ZeroOperator(g) for g in grid] for _ in grid]
        super().__init__(name, entries)


class Identity(Operator):
    def __init__(self, grid: [base.Grid], name='I'):
        entries = []
        for i, _ in enumerate(grid):
            entries.append([])
            for j, g in enumerate(grid):
                if i == j:
                    entries[i].append(base.Identity(g))
                else:
                    entries[i].append(base.ZeroOperator(g))
        super().__init__(name, entries)


class Approximation(base.Entity):

    def __init__(self, name, entries):
        self._name = name
        self._entries = entries
        self._shape = (functools.reduce(lambda a, b: a.shape[0] + b.shape[0], entries), entries[0].shape[1])
        super().__init__()

    @property
    def entries(self):
        return self._entries

    @property
    def grid(self):
        return list(map(lambda entry: entry.grid, self.entries))


class RightHandSide(Approximation):
    pass


class ZeroApproximation(Approximation):
    def __init__(self, grid: [base.Grid], name='0'):
        super().__init__(name, [base.ZeroApproximation(g) for g in grid])


class InterGridOperator(Operator):
    def __init__(self, name, fine_grid: [base.Grid], coarse_grid: [base.Grid], stencil_generator,
                 InterGridOperatorType, ZeroOperatorType):
        self._stencil_generator = stencil_generator
        entries = [[InterGridOperatorType(name, fg, cg, stencil_generator)
                    if i == j else ZeroOperatorType(fg, cg) for j in range(len(fine_grid))]
                   for i, (fg, cg) in enumerate(zip(fine_grid, coarse_grid))]
        super().__init__(name, entries)

    @property
    def stencil_generator(self):
        return self._stencil_generator


class Restriction(InterGridOperator):
    def __init__(self, name, fine_grid: [base.Grid], coarse_grid: [base.Grid], stencil_generator=None):
        super().__init__(name, fine_grid, coarse_grid, stencil_generator,
                         multigrid.Restriction, multigrid.ZeroRestriction)


class Prolongation(InterGridOperator):
    def __init__(self, name, fine_grid: [base.Grid], coarse_grid: [base.Grid], stencil_generator=None):
        super().__init__(name, fine_grid, coarse_grid, stencil_generator,
                         multigrid.Prolongation, multigrid.ZeroProlongation)


class Diagonal(base.UnaryExpression):
    pass


class ElementwiseDiagonal(base.UnaryExpression):
    pass


def get_coarse_grid(grid: [base.Grid], coarsening_factor):
    return list(map(lambda g: multigrid.get_coarse_grid(g, coarsening_factor), grid))


def get_coarse_approximation(approximation: Approximation, coarsening_factor):
    return Approximation(f'{approximation.name}_c', get_coarse_grid(approximation.grid, coarsening_factor))


def get_coarse_rhs(rhs: RightHandSide, coarsening_factor):
    return RightHandSide(f'{rhs.name}_c', get_coarse_grid(rhs.grid, coarsening_factor))


def get_coarse_operator(operator, coarse_grid):
    new_entries = [[base.Operator(f'{entry.name}_c', coarse_grid[i], entry.stencil_generator) for entry in row]
                   for i, row in enumerate(operator.entries)]
    return Operator(f'{operator.name}_c', new_entries)