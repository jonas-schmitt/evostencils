from evostencils.expressions import base


class Restriction(base.Operator):
    def __init__(self, grid, coarse_grid, stencil=None):
        self._grid = grid
        self._coarse_grid = coarse_grid
        super(Restriction, self).__init__(f'R_{grid.size}', (coarse_grid.size, grid.size), stencil)

    def __repr__(self):
        return f'Restriction({repr(self._grid)}, {repr(self._coarse_grid)}, {repr(self.generate_stencil())})'


class Interpolation(base.Operator):
    def __init__(self, grid, coarse_grid, stencil=None):
        self._grid = grid
        self._coarse_grid = coarse_grid
        super(Interpolation, self).__init__(f'I_{coarse_grid.size}', (grid.size, coarse_grid.size), stencil)

    def __repr__(self):
        return f'Interpolation({repr(self._grid)}, {repr(self._coarse_grid)}, {repr(self.generate_stencil())})'


class CoarseGridSolver(base.Operator):
    def __init__(self, coarse_grid):
        self._grid = coarse_grid
        super(CoarseGridSolver, self).__init__(f'S_{coarse_grid.size}', (coarse_grid.size, coarse_grid.size), None)

    def __repr__(self):
        return f'CoarseGridSolver({repr(self._coarse_grid)})'


class Correction(base.Expression):
    def __init__(self, iteration_matrix: base.Expression, grid, operator: base.Operator, rhs: base.Grid, weight=0.5):
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
        return base.Addition(base.Scaling(1 - self.weight, u), base.Scaling(self.weight, base.Multiplication(B, residual(u, A, f))))

    def __repr__(self):
        return f'Correction({repr(self.iteration_matrix)}, {repr(self.grid)}, {repr(self.operator)}, {repr(self.rhs)}, {repr(self.weight)})'

    def __str__(self):
        return str(self.generate_expression())


def correct(operator, rhs, iteration_matrix, grid, weight=0.5):
    return Correction(iteration_matrix, grid, operator, rhs, weight)


def residual(grid, operator, rhs):
    return base.Subtraction(rhs, base.Multiplication(operator, grid))


def get_interpolation(grid, coarse_grid, stencil=None):
    return Interpolation(grid, coarse_grid, stencil)


def get_restriction(grid, coarse_grid, stencil=None):
    return Restriction(grid, coarse_grid, stencil)


def get_coarse_grid(grid, coarsening_factor):
    return base.Grid(f'{grid.name}_coarse', grid.size / coarsening_factor)


def get_coarse_operator(operator, coarsening_factor, stencil=None):
    return base.Operator(f'{operator.name}_coarse', (operator.shape[0] / coarsening_factor,
                                                     operator.shape[1] / coarsening_factor), stencil)


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