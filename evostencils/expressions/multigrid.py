from evostencils.expressions import base
from evostencils.expressions import partitioning as part


class Restriction(base.Operator):
    def __init__(self, grid, coarse_grid, stencil=None):
        self._grid = grid
        self._coarse_grid = coarse_grid
        super(Restriction, self).__init__(f'R_{grid.shape[0]}', (coarse_grid.shape[0], grid.shape[0]), stencil)

    @property
    def grid(self):
        return self._grid

    @property
    def coarse_grid(self):
        return self._coarse_grid

    def __repr__(self):
        return f'Restriction({repr(self._grid)}, {repr(self._coarse_grid)}, {repr(self.generate_stencil())})'


class Interpolation(base.Operator):
    def __init__(self, grid, coarse_grid, stencil=None):
        self._grid = grid
        self._coarse_grid = coarse_grid
        super(Interpolation, self).__init__(f'I_{coarse_grid.shape[0]}', (grid.shape[0], coarse_grid.shape[0]), stencil)

    @property
    def grid(self):
        return self._grid

    @property
    def coarse_grid(self):
        return self._coarse_grid

    def __repr__(self):
        return f'Interpolation({repr(self._grid)}, {repr(self._coarse_grid)}, {repr(self.generate_stencil())})'


class CoarseGridSolver(base.Operator):
    def __init__(self, grid, operator):
        self._grid = grid
        self._operator = operator
        super(CoarseGridSolver, self).__init__(f'S_{grid.shape[0]}',
                                               (grid.shape[0], grid.shape[0]), None)

    @property
    def grid(self):
        return self._grid

    @property
    def operator(self):
        return self._operator

    def __repr__(self):
        return f'CoarseGridSolver({repr(self._grid)})'


class Cycle(base.Expression):
    def __init__(self, correction: base.Expression, grid: base.Expression, partitioning=part.Single, weight=1.0):
        self._correction = correction
        self._grid = grid
        self._weight = weight
        self._partitioning = partitioning

    @property
    def shape(self):
        return self._grid.shape

    @property
    def stencil(self):
        return None

    @property
    def correction(self):
        return self._correction

    @property
    def grid(self):
        return self._grid

    @property
    def weight(self):
        return self._weight

    @property
    def partitioning(self):
        return self._partitioning

    def generate_expression(self):
        return base.Addition(self.grid, base.Scaling(self.weight, self.correction))

    def __repr__(self):
        return f'Cycle({repr(self.correction)}, {repr(self.grid)}, {repr(self.partitioning)}, {repr(self.weight)}'

    def __str__(self):
        return str(self.generate_expression())

    def apply(self, transform: callable, *args):
        correction = transform(self.correction, *args)
        grid = transform(self.grid, *args)
        return Cycle(correction, grid, self.partitioning, self.weight)


def cycle(grid, correction, partitioning=part.Single, weight=1.0):
    return Cycle(correction, grid, partitioning, weight)


def residual(grid, operator, rhs):
    return base.Subtraction(rhs, base.Multiplication(operator, grid))


def get_interpolation(grid: base.Grid, coarse_grid: base.Grid, stencil=None):
    return Interpolation(grid, coarse_grid, stencil)


def get_restriction(grid: base.Grid, coarse_grid: base.Grid, stencil=None):
    return Restriction(grid, coarse_grid, stencil)


def get_coarse_grid(grid: base.Grid, coarsening_factor: tuple):
    from functools import reduce
    import operator
    coarse_size = tuple(grid.size[i] // coarsening_factor[i] for i in range(len(grid.size)))
    return base.Grid(f'{grid.name}_{reduce(operator.mul, coarse_size)}', coarse_size)


def get_coarse_operator(operator, coarse_grid, stencil=None):
    if stencil is None:
        stencil = operator.generate_stencil()
    return base.Operator(f'{operator.name}_{coarse_grid.shape[0]}',
                         (coarse_grid.shape[0], coarse_grid.shape[0]), stencil)


def get_coarse_grid_solver(coarse_grid):
    return CoarseGridSolver(coarse_grid)


def is_intergrid_operation(expression: base.Expression) -> bool:
    return isinstance(expression, Restriction) or isinstance(expression, Interpolation) \
           or isinstance(expression, CoarseGridSolver)

