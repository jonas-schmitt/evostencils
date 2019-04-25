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


class IntergridOperator(Operator):
    def __init__(self, name, scalar_intergrid_operator, grid: [base.Grid]):
        number_of_unknowns = len(grid)
        entries = [[] for _ in range(number_of_unknowns)]
        for i in range(number_of_unknowns):
            for j in range(number_of_unknowns):
                if i == j:
                    entries[i].append(scalar_intergrid_operator)
                else:
                    entries[j].append(base.ZeroOperator(grid[j], shape=scalar_intergrid_operator.shape))
        super().__init__(name, entries)


class Restriction(IntergridOperator):
    def __init__(self, name, scalar_restriction_operator, grid: [base.Grid]):
        super().__init__(name, scalar_restriction_operator, grid)


class Prolongation(IntergridOperator):
    def __init__(self, name, scalar_prolongation_operator, grid: [base.Grid]):
        super().__init__(name, scalar_prolongation_operator, grid)


def get_coarse_grid(grid: [base.Grid], coarsening_factor):
    return list(map(lambda g: multigrid.get_coarse_grid(g, coarsening_factor), grid))

