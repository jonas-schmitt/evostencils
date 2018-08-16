from evostencils.expressions import base


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
    def __init__(self, coarse_grid):
        self._grid = coarse_grid
        super(CoarseGridSolver, self).__init__(f'S_{coarse_grid.shape[0]}', (coarse_grid.shape[0], coarse_grid.shape[0]), None)

    @property
    def grid(self):
        return self._grid

    def __repr__(self):
        return f'CoarseGridSolver({repr(self._grid)})'


class Correction(base.Expression):
    def __init__(self, iteration_matrix: base.Expression, grid: base.Expression, operator: base.Expression, rhs: base.Expression, weight=1.0):
        self._iteration_matrix = iteration_matrix
        self._grid = grid
        self._operator = operator
        self._rhs = rhs
        self._weight = weight

    @property
    def shape(self):
        return self._grid.shape

    @property
    def stencil(self):
        return None

    @property
    def iteration_matrix(self):
        return self._iteration_matrix

    @property
    def grid(self):
        return self._grid

    @property
    def operator(self):
        return self._operator

    @property
    def rhs(self):
        return self._rhs

    @property
    def weight(self):
        return self._weight

    def generate_expression(self):
        A = self.operator
        u = self.grid
        f = self.rhs
        B = self.iteration_matrix
        return base.Addition(u, base.Scaling(self.weight, base.Multiplication(B, residual(u, A, f))))

    def __repr__(self):
        return f'Correction({repr(self.iteration_matrix)}, {repr(self.grid)}, {repr(self.operator)}, {repr(self.rhs)}, {repr(self.weight)})'

    def __str__(self):
        return str(self.generate_expression())

    def apply(self, transform: callable):
        iteration_matrix = transform(self.iteration_matrix)
        grid = transform(self.grid)
        operator = transform(self.operator)
        rhs = transform(self.rhs)
        return Correction(iteration_matrix, grid, operator, rhs, self.weight)


def correct(operator, rhs, iteration_matrix, grid, weight=1.0):
    return Correction(iteration_matrix, grid, operator, rhs, weight)


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
    return base.Operator(f'{operator.name}_{coarse_grid.shape[0]}', (coarse_grid.shape[0], coarse_grid.shape[0]), stencil)


def get_coarse_grid_solver(coarse_grid):
    return CoarseGridSolver(coarse_grid)


def is_intergrid_operation(expression: base.Expression) -> bool:
    return isinstance(expression, Restriction) or isinstance(expression, Interpolation) \
           or isinstance(expression, CoarseGridSolver)


def contains_intergrid_operation(expression: base.Expression) -> bool:
    if isinstance(expression, Restriction) or isinstance(expression.Interpolation):
        return True
    elif isinstance(expression, base.Entity):
        return False
    elif isinstance(expression, base.UnaryExpression):
        return contains_intergrid_operation(expression.operand)
    elif isinstance(expression, base.BinaryExpression):
        return contains_intergrid_operation(expression.operand1) or contains_intergrid_operation(expression.operand2)
    elif isinstance(expression, base.Scaling):
        return contains_intergrid_operation(expression.operand)
    else:
        raise NotImplementedError("Not implemented")