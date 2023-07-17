from evostencils.ir import base
from typing import List, Tuple

class System(base.Expression):

    def __init__(self, name, entries, shape):
        self._name = name
        self._entries = entries
        self._shape = shape
        self.smoother_type=None
        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def entries(self):
        return self._entries

    @property
    def shape(self):
        return self._shape

    def apply(self, _, *args):
        return self

    def mutate(self, _, *args):
        pass


class Operator(System):

    def __init__(self, name, entries):
        shape = [0, 0]
        for row in entries:
            entry = row[0]
            shape[0] += entry.shape[0]
        for entry in entries[0]:
            shape[1] += entry.shape[1]
        shape = tuple(shape)
        super().__init__(name, entries, shape)

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


class Approximation(System):

    def __init__(self, name, entries):
        if len(entries) == 1:
            shape = entries[0].shape
        else:
            acc = 0
            for entry in entries:
                acc += entry.shape[0]
            shape = tuple((acc, entries[0].shape[1]))
        super().__init__(name, entries, shape)

    @property
    def entries(self):
        return self._entries

    @property
    def grid(self):
        return list(map(lambda entry: entry.grid, self.entries))

    @property
    def predecessor(self):
        return None


class RightHandSide(Approximation):
    pass


class ZeroApproximation(Approximation):
    def __init__(self, grid: [base.Grid], name='0'):
        super().__init__(name, [base.ZeroApproximation(g) for g in grid])


class InterGridOperator(Operator):
    def __init__(self, name, list_of_intergrid_operators, ZeroOperatorType):
        entries = [[intergrid_operator if i == j else ZeroOperatorType(intergrid_operator.fine_grid, intergrid_operator.coarse_grid)
                    for j in range(len(list_of_intergrid_operators))] for i, intergrid_operator in enumerate(list_of_intergrid_operators)]
        super().__init__(name, entries)


class Restriction(InterGridOperator):
    def __init__(self, name, list_of_intergrid_operators):
        super().__init__(name, list_of_intergrid_operators, base.ZeroRestriction)


class Prolongation(InterGridOperator):
    def __init__(self, name, list_of_intergrid_operators):
        super().__init__(name, list_of_intergrid_operators, base.ZeroProlongation)


class Diagonal(base.UnaryExpression):
    pass


class ElementwiseDiagonal(base.UnaryExpression):
    def __str__(self):
        return "D"


class Jacobian(base.UnaryExpression):
    def __init__(self, dummy_op, n_newton_steps):
        self.n_newton_steps = n_newton_steps
        super().__init__(dummy_op)

    def __str__(self):
        return f"J[{self.n_newton_steps}]"


def get_coarse_grid(grid: [base.Grid], coarsening_factors: List[Tuple[int, ...]]):
    return [base.get_coarse_grid(g, cf) for g, cf in zip(grid, coarsening_factors)]


def get_coarse_approximation(approximation: Approximation, coarsening_factors: List[Tuple[int, ...]]):
    return Approximation(f'{approximation.name}', [base.Approximation(f'{entry.name}_c', base.get_coarse_grid(entry.grid, cf))
                                                   for entry, cf in zip(approximation.entries, coarsening_factors)])


def get_coarse_rhs(rhs: RightHandSide, coarsening_factors: List[Tuple[int, ...]]):
    return RightHandSide(f'{rhs.name}', [base.RightHandSide(f'{entry.name}_c', base.get_coarse_grid(entry.grid, cf))
                                         for entry, cf in zip(rhs.entries, coarsening_factors)])


def get_coarse_operator(operator, coarse_grid):
    new_entries = [[base.Operator(f'{entry.name}_c', coarse_grid[i], entry.stencil_generator) for entry in row]
                   for i, row in enumerate(operator.entries)]
    return Operator(f'{operator.name}', new_entries)
