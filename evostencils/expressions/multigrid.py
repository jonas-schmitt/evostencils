from evostencils.expressions import base
from evostencils.expressions import partitioning as part
from evostencils.stencils import gallery


class InterGridOperator(base.Operator):

    def __init__(self, name, grid, fine_grid, coarse_grid, stencil_generator):
        self._fine_grid = fine_grid
        self._coarse_grid = coarse_grid
        super().__init__(name, grid, stencil_generator)

    @property
    def fine_grid(self):
        return self._fine_grid

    @property
    def coarse_grid(self):
        return self._coarse_grid


class Restriction(InterGridOperator):
    def __init__(self, name, fine_grid, coarse_grid, stencil_generator=None):
        super().__init__(name, coarse_grid, fine_grid, coarse_grid, stencil_generator)
        import operator
        from functools import reduce
        tmp1 = reduce(operator.mul, fine_grid.size)
        tmp2 = reduce(operator.mul, coarse_grid.size)
        self._shape = (tmp2, tmp1)

    @property
    def fine_grid(self):
        return self._fine_grid

    @property
    def coarse_grid(self):
        return self._coarse_grid

    def __repr__(self):
        return f'Restriction({repr(self.name)}, {repr(self.fine_grid)}, ' \
               f'{repr(self.coarse_grid)}, {repr(self.generate_stencil())})'


class ZeroRestriction(Restriction):
    def __init__(self, fine_grid, coarse_grid, name='0'):
        super().__init__(name, fine_grid, coarse_grid, gallery.ZeroGenerator)


class Prolongation(InterGridOperator):
    def __init__(self, name, fine_grid, coarse_grid, stencil_generator=None):
        super().__init__(name, fine_grid, fine_grid, coarse_grid, stencil_generator)
        import operator
        from functools import reduce
        tmp1 = reduce(operator.mul, fine_grid.size)
        tmp2 = reduce(operator.mul, coarse_grid.size)
        self._shape = (tmp1, tmp2)

    @property
    def fine_grid(self):
        return self._fine_grid

    @property
    def coarse_grid(self):
        return self._coarse_grid

    def __repr__(self):
        return f'Interpolation({repr(self.name)}, {repr(self.fine_grid)}, ' \
               f'{repr(self.coarse_grid)}, {repr(self.generate_stencil())})'


class ZeroProlongation(Prolongation):
    def __init__(self, fine_grid, coarse_grid, name='0'):
        super().__init__(name, fine_grid, coarse_grid, gallery.ZeroGenerator)


class CoarseGridSolver(base.Entity):
    def __init__(self, operator, expression=None):
        self._name = "CGS"
        self._shape = operator.shape
        self._operator = operator
        self._expression = expression
        super().__init__()

    @staticmethod
    def generate_stencil():
        return None

    @property
    def grid(self):
        return self.operator.grid

    @property
    def operator(self):
        return self._operator

    @property
    def expression(self):
        return self._expression

    def __repr__(self):
        return f'CoarseGridSolver({repr(self.operator)}, {repr(self.expression)})'

    def mutate(self, f: callable, *args):
        f(self.operator, *args)


class Residual(base.Expression):
    def __init__(self, operator, approximation, rhs):
        # assert iterate.shape == rhs.shape, "Shapes of iterate and rhs must match"
        self._operator = operator
        self._approximation = approximation
        self._rhs = rhs
        super().__init__()

    @property
    def shape(self):
        return self.approximation.shape

    @property
    def grid(self):
        return self.approximation.grid

    @property
    def operator(self):
        return self._operator

    @property
    def approximation(self):
        return self._approximation

    @property
    def rhs(self):
        return self._rhs

    @staticmethod
    def generate_stencil():
        return None

    def generate_expression(self):
        return base.sub(self.rhs, base.mul(self.operator, self.approximation))

    def __str__(self):
        return f'({str(self.rhs)} - {str(self.operator)} * {str(self.approximation)})'

    def __repr__(self):
        return f'Residual({repr(self.operator)}, {repr(self.approximation)}, {repr(self.rhs)})'

    def apply(self, transform: callable, *args):
        operator = transform(self.operator, *args)
        iterate = transform(self.approximation, *args)
        rhs = transform(self.rhs, *args)
        return Residual(operator, iterate, rhs)

    def mutate(self, f: callable, *args):
        # We already mutate within the cycle node
        pass


class Cycle(base.Expression):
    def __init__(self, approximation, rhs, correction, partitioning=part.Single, weight=1.0, predecessor=None):
        # assert iterate.shape == correction.shape, "Shapes must match"
        # assert iterate.grid.size == correction.grid.size and iterate.grid.step_size == correction.grid.step_size, \
        #    "Grids must match"
        self._approximation = approximation
        self._rhs = rhs
        self._correction = correction
        self._weight = weight
        self._partitioning = partitioning
        self.predecessor = predecessor
        self.global_id = None
        self.weight_obtained = False
        self.weight_set = False
        super().__init__()

    @property
    def shape(self):
        return self._approximation.shape

    @property
    def grid(self):
        return self.approximation.grid

    @property
    def approximation(self):
        return self._approximation

    @property
    def rhs(self):
        return self._rhs

    @property
    def correction(self):
        return self._correction

    @property
    def weight(self):
        return self._weight

    @property
    def partitioning(self):
        return self._partitioning

    @staticmethod
    def generate_stencil():
        return None

    def generate_expression(self):
        return base.Addition(self.approximation, base.Scaling(self.weight, self.correction))

    def __repr__(self):
        return f'Cycle({repr(self.approximation)}, {repr(self.rhs)}, {repr(self.correction)}, ' \
               f'{repr(self.partitioning)}, {repr(self.weight)}'

    def __str__(self):
        return str(self.generate_expression())

    def apply(self, transform: callable, *args):
        approximation = transform(self.approximation, *args)
        rhs = transform(self.rhs, *args)
        correction = transform(self.correction, *args)
        return Cycle(approximation, rhs, correction, self.partitioning, self.weight, self.predecessor)

    def mutate(self, f: callable, *args):
        f(self.approximation, *args)
        f(self.rhs, *args)
        f(self.correction, *args)


def cycle(iterate, rhs, correction, partitioning=part.Single, weight=1, predecessor=None):
    return Cycle(iterate, rhs, correction, partitioning, weight, predecessor)


def residual(operator, iterate, rhs):
    # return base.Subtraction(rhs, base.Multiplication(operator, iterate))
    return Residual(operator, iterate, rhs)


def get_coarse_grid(grid: base.Grid, coarsening_factor):
    coarse_size = tuple(size // factor for size, factor in zip(grid.size, coarsening_factor))
    coarse_step_size = tuple(h * factor for h, factor in zip(grid.step_size, coarsening_factor))
    return base.Grid(coarse_size, coarse_step_size)


def get_coarse_approximation(approximation: base.Approximation, coarsening_factor):
    return base.Approximation(f'{approximation.name}_c', get_coarse_grid(approximation.grid, coarsening_factor))


def get_coarse_rhs(rhs: base.RightHandSide, coarsening_factor):
    return base.RightHandSide(f'{rhs.name}_c', get_coarse_grid(rhs.grid, coarsening_factor))


def get_coarse_operator(operator, coarse_grid):
    return base.Operator(f'{operator.name}', coarse_grid, operator.stencil_generator)


def is_intergrid_operation(expression: base.Expression) -> bool:
    return isinstance(expression, Restriction) or isinstance(expression, Prolongation) \
           or isinstance(expression, CoarseGridSolver)


def contains_intergrid_operation(expression: base.Expression) -> bool:
    if isinstance(expression, base.BinaryExpression):
        return contains_intergrid_operation(expression.operand1) or contains_intergrid_operation(expression.operand2)
    elif isinstance(expression, base.UnaryExpression):
        return contains_intergrid_operation(expression.operand)
    elif isinstance(expression, base.Scaling):
        return contains_intergrid_operation(expression.operand)
    elif isinstance(expression, CoarseGridSolver):
        return True
    elif isinstance(expression, base.Operator):
        return is_intergrid_operation(expression)
    else:
        raise RuntimeError("Expression does not only contain operators")


def determine_maximum_tree_depth(expression: base.Expression) -> int:
    if isinstance(expression, base.Entity):
        return 0
    elif isinstance(expression, base.UnaryExpression) or isinstance(expression, base.Scaling):
        return determine_maximum_tree_depth(expression.operand) + 1
    elif isinstance(expression, base.BinaryExpression):
        return max(determine_maximum_tree_depth(expression.operand1), determine_maximum_tree_depth(expression.operand2)) + 1
    elif isinstance(expression, Residual):
        return max(determine_maximum_tree_depth(expression.rhs), determine_maximum_tree_depth(expression.approximation) + 1) + 1
    elif isinstance(expression, Cycle):
        return max(determine_maximum_tree_depth(expression.approximation), determine_maximum_tree_depth(expression.correction) + 1) + 1
    else:
        raise RuntimeError("Case not implemented")
