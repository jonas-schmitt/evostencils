import abc
from functools import reduce
from evostencils.ir import partitioning as part
from evostencils.stencils import multiple, gallery
from evostencils.stencils import constant


# Base classes
class Expression(abc.ABC):
    def __init__(self):
        self.lfa_symbol = None
        self.valid = False
        self.runtime = None

    @property
    @abc.abstractmethod
    def shape(self):
        pass

    @property
    @abc.abstractmethod
    def grid(self):
        pass

    @abc.abstractmethod
    def apply(self, transform: callable, *args):
        pass

    @abc.abstractmethod
    def mutate(self, f: callable, *args):
        pass


class Entity(Expression):

    def __init__(self, name, grid, shape):
        self._name = name
        self._grid = grid
        self._shape = shape
        super().__init__()

    @property
    def name(self):
        return self._name

    @property
    def grid(self):
        return self._grid

    @property
    def shape(self):
        return self._shape

    def __str__(self):
        return f'{self.name}'

    def apply(self, _, *args):
        return self

    def mutate(self, _, *args):
        pass


class UnaryExpression(Expression):

    def __init__(self, operand):
        self._operand = operand
        self._shape = operand.shape
        super().__init__()

    @property
    def operand(self):
        return self._operand

    @property
    def shape(self):
        return self._shape

    def apply(self, transform: callable, *args):
        return type(self)(transform(self.operand, *args))

    def mutate(self, f: callable, *args):
        f(self.operand, *args)

    @property
    def grid(self):
        return self.operand.grid


class BinaryExpression(Expression):

    def __init__(self, operand1, operand2):
        self._operand1 = operand1
        self._operand2 = operand2
        super().__init__()

    @property
    def operand1(self):
        return self._operand1

    @property
    def operand2(self):
        return self._operand2

    @property
    def shape(self):
        raise NotImplementedError("Shape undefined in binary expression")

    @property
    def grid(self):
        return self.operand1.grid

    def apply(self, transform: callable, *args):
        return type(self)(transform(self.operand1, *args), transform(self.operand2, *args))

    def mutate(self, f: callable, *args):
        f(self.operand1, *args)
        f(self.operand2, *args)


# Entities
class Operator(Entity):
    def __init__(self, name, grid, stencil_generator=None):
        import operator
        tmp = reduce(operator.mul, grid.size)
        shape = (tmp, tmp)
        self._stencil_generator = stencil_generator
        super().__init__(name, grid, shape)

    @property
    def stencil_generator(self):
        return self._stencil_generator

    def generate_stencil(self):
        if self._stencil_generator is None:
            return None
        return self._stencil_generator.generate_stencil(self._grid)

    def generate_exa3(self):
        if self._stencil_generator is None:
            return None
        return self._stencil_generator.generate_exa3(self._name)

    def __repr__(self):
        return f'Operator({repr(self.name)}, {repr(self.grid)}, {repr(self._stencil_generator)})'


class Identity(Operator):
    def __init__(self, grid, name='I'):
        from evostencils.stencils.gallery import IdentityGenerator
        super().__init__(name, grid, IdentityGenerator(grid.dimension))

    def __repr__(self):
        return f'Identity({repr(self.grid)})'


class ZeroOperator(Operator):
    def __init__(self, grid, shape=None, name='0'):
        from evostencils.stencils.gallery import ZeroGenerator
        super().__init__(name, grid, ZeroGenerator(grid.dimension))
        if shape is not None:
            self._shape = shape

    def __repr__(self):
        return f'ZeroOperator({repr(self.grid)})'


class Grid:
    def __init__(self, size, spacing, level):
        assert len(size) == len(spacing), "Dimensions of the size and step size must match"
        self._size = size
        self._spacing = spacing
        self._level = level

    @property
    def size(self):
        return self._size

    @property
    def spacing(self):
        return self._spacing

    @property
    def level(self):
        return self._level

    @property
    def dimension(self):
        return len(self.size)

    def __eq__(self, other):
        if isinstance(other, Grid):
            return self.size == other.size and self.spacing == other.spacing

    def __repr__(self):
        return f'Grid({repr(self.size)}, {repr(self.spacing)}'


class Approximation(Entity):
    def __init__(self, name, grid):
        import operator
        shape = (reduce(operator.mul, grid.size), 1)
        super().__init__(name, grid, shape)

    def __eq__(self, other):
        if not isinstance(other, Approximation):
            return False
        return self.name == other.name and self.grid == other.grid

    @property
    def predecessor(self):
        return None

    def generate_stencil(self):
        return constant.get_unit_stencil(self)

    def __repr__(self):
        return f'Approximation({repr(self.name)}, {repr(self.grid)})'


class RightHandSide(Approximation):

    def __eq__(self, other):
        if isinstance(other, RightHandSide):
            return self == other
        else:
            return False

    def generate_stencil(self):
        return constant.get_null_stencil(self)

    def __repr__(self):
        return f'RightHandSide({repr(self.name)}, {repr(self.grid)})'


class ZeroApproximation(Approximation):

    def generate_stencil(self):
        return constant.get_null_stencil(self)

    def __init__(self, grid, name='0'):
        super(ZeroApproximation, self).__init__(name, grid)

    def __repr__(self):
        return f'ZeroApproximation({repr(self.grid)})'


# Unary Expressions
class Diagonal(UnaryExpression):
    def generate_stencil(self):
        return multiple.diagonal(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.diag'

    def __repr__(self):
        return f'Diagonal({repr(self.operand)})'


class LowerTriangle(UnaryExpression):
    def generate_stencil(self):
        return multiple.lower(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.lower'

    def __repr__(self):
        return f'LowerTriangle({repr(self.operand)})'


class UpperTriangle(UnaryExpression):
    def generate_stencil(self):
        return multiple.upper(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.upper'

    def __repr__(self):
        return f'UpperTriangle({repr(self.operand)})'


class BlockDiagonal(UnaryExpression):
    def __init__(self, operand, block_size):
        self._block_size = block_size
        super().__init__(operand)

    def generate_stencil(self):
        return multiple.block_diagonal(self.operand.generate_stencil(), self.block_size)

    @property
    def block_size(self):
        return self._block_size

    def __str__(self):
        return f'{str(self.operand)}.block_diag'

    def __repr__(self):
        return f'BlockDiagonal({repr(self.operand)}, {repr(self.block_size)})'

    def apply(self, transform: callable, *args):
        return type(self)(transform(self.operand, *args), self.block_size)


class Inverse(UnaryExpression):
    def generate_stencil(self):
        return multiple.inverse(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.I'

    def __repr__(self):
        return f'Inverse({repr(self.operand)})'


class Transpose(UnaryExpression):
    def __init__(self, operand):
        self._operand = operand
        self._shape = (operand.shape[1], operand.shape[0])
        super().__init__(operand)

    def generate_stencil(self):
        return multiple.transpose(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.T'

    def __repr__(self):
        return f'Transpose({repr(self.operand)})'


# Binary Expressions
class Addition(BinaryExpression):

    def __init__(self, operand1, operand2):
        # assert operand1.shape == operand2.shape, "Operand shapes are not equal"
        # assert operand1.grid.size == operand2.grid.size and operand1.grid.step_size == operand2.grid.step_size, \
        #     "Grids must match"
        self._shape = operand1.shape
        super().__init__(operand1, operand2)

    @property
    def grid(self):
        return self.operand1.grid

    @property
    def shape(self):
        return self._shape

    def generate_stencil(self):
        return multiple.add(self.operand1.generate_stencil(), self.operand2.generate_stencil())

    def __str__(self):
        return f'({str(self.operand1)} + {str(self.operand2)})'

    def __repr__(self):
        return f'Addition({repr(self.operand1)}, {repr(self.operand2)})'


class Subtraction(BinaryExpression):

    def __init__(self, operand1, operand2):
        # assert operand1.shape == operand2.shape, "Operand shapes are not equal"
        # assert operand1.grid.size == operand2.grid.size and operand1.grid.step_size == operand2.grid.step_size, \
        #     "Grids must match"
        self._shape = operand1.shape
        super().__init__(operand1, operand2)

    @property
    def grid(self):
        return self.operand1.grid

    @property
    def shape(self):
        return self._shape

    def generate_stencil(self):
        return multiple.sub(self.operand1.generate_stencil(), self.operand2.generate_stencil())

    def __str__(self):
        return f'({str(self.operand1)} - {str(self.operand2)})'

    def __repr__(self):
        return f'Subtraction({repr(self.operand1)}, {repr(self.operand2)})'


class Multiplication(BinaryExpression):

    def __init__(self, operand1, operand2):
        assert operand1.shape[1] == operand2.shape[0], "Operand shapes are not aligned"
        self._shape = (operand1.shape[0], operand2.shape[1])
        super().__init__(operand1, operand2)

    @property
    def grid(self):
        return self.operand1.grid

    @property
    def shape(self):
        return self._shape

    def generate_stencil(self):
        return multiple.mul(self.operand1.generate_stencil(), self.operand2.generate_stencil())

    def __str__(self):
        return f'({str(self.operand1)} * {str(self.operand2)})'

    def __repr__(self):
        return f'Multiplication({repr(self.operand1)}, {repr(self.operand2)})'


# Scaling
class Scaling(Expression):

    def __init__(self, factor, operand):
        self._factor = factor
        self._operand = operand
        self._shape = operand.shape
        super().__init__()

    @property
    def factor(self):
        return self._factor

    @property
    def operand(self):
        return self._operand

    @property
    def grid(self):
        return self.operand.grid

    @property
    def shape(self):
        return self._shape

    def generate_stencil(self):
        return multiple.scale(self.factor, self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.factor)} * {str(self.operand)}'

    def __repr__(self):
        return f'Scaling({repr(self.factor)}, {repr(self.operand)})'

    def apply(self, transform: callable, *args):
        return Scaling(self.factor, transform(self.operand, *args))

    def mutate(self, f: callable, *args):
        f(self.operand, *args)


# Wrapper functions

def diag(operand):
    return Diagonal(operand)


def inv(operand):
    return Inverse(operand)


def add(operand1, operand2):
    return Addition(operand1, operand2)


def sub(operand1, operand2):
    return Subtraction(operand1, operand2)


def mul(operand1, operand2):
    return Multiplication(operand1, operand2)


def scale(factor, operand):
    return Scaling(factor, operand)


def minus(operand):
    return Scaling(-1, operand)


def is_quadratic(expression: Expression) -> bool:
    return expression.shape[0] == expression.shape[1]


class InterGridOperator(Operator):

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

    @property
    def input_grid(self):
        return self.fine_grid

    @property
    def output_grid(self):
        return self.coarse_grid

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

    @property
    def input_grid(self):
        return self.coarse_grid

    @property
    def output_grid(self):
        return self.fine_grid

    def __repr__(self):
        return f'Interpolation({repr(self.name)}, {repr(self.fine_grid)}, ' \
               f'{repr(self.coarse_grid)}, {repr(self.generate_stencil())})'


class ZeroProlongation(Prolongation):
    def __init__(self, fine_grid, coarse_grid, name='0'):
        super().__init__(name, fine_grid, coarse_grid, gallery.ZeroGenerator)


class CoarseGridSolver(Entity):
    def __init__(self, name, operator, expression=None):
        shape = operator.shape
        self._operator = operator
        self._expression = expression
        self.additional_info = {}
        super().__init__(name, operator.grid, shape)

    @staticmethod
    def generate_stencil():
        return None

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


class Residual(Expression):
    def __init__(self, operator, approximation, rhs):
        # assert iterate.shape == rhs.shape, "Shapes of iterate and rhs must match"
        self._operator = operator
        self._approximation = approximation
        self._rhs = rhs
        super().__init__()

    @property
    def shape(self):
        return self.rhs.shape

    @property
    def grid(self):
        return self.rhs.grid

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
        return sub(self.rhs, mul(self.operator, self.approximation))

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
        f(self.rhs, *args)
        f(self.approximation, *args)


class Cycle(Expression):
    def __init__(self, approximation, rhs, correction=None, partitioning=part.Single,
                 relaxation_factor=1.0, predecessor=None):
        # assert iterate.shape == correction.shape, "Shapes must match"
        # assert iterate.grid.size == correction.grid.size and iterate.grid.step_size == correction.grid.step_size, \
        #    "Grids must match"
        self.approximation = approximation
        self.rhs = rhs
        self.correction = correction
        self.relaxation_factor = relaxation_factor
        self.additional_info = {}
        self.partitioning = partitioning
        self.predecessor = predecessor
        self.global_id = None
        self.weight_obtained = False
        self.weight_set = False
        super().__init__()

    @property
    def shape(self):
        return self.approximation.shape

    @property
    def grid(self):
        return self.approximation.grid

    @staticmethod
    def generate_stencil():
        return None

    def generate_expression(self):
        return Addition(self.approximation, Scaling(self.relaxation_factor, self.correction))

    def __repr__(self):
        return f'Cycle({repr(self.approximation)}, {repr(self.rhs)}, {repr(self.correction)}, ' \
               f'{repr(self.partitioning)}, {repr(self.relaxation_factor)}'

    def __str__(self):
        return str(self.generate_expression())

    def apply(self, transform: callable, *args):
        approximation = transform(self.approximation, *args)
        rhs = transform(self.rhs, *args)
        correction = transform(self.correction, *args)
        return Cycle(approximation, rhs, correction, self.partitioning, self.relaxation_factor, self.predecessor)

    def mutate(self, f: callable, *args):
        f(self.correction, *args)


def get_coarse_grid(grid: Grid, coarsening_factor):
    coarse_size = tuple(size // factor for size, factor in zip(grid.size, coarsening_factor))
    coarse_step_size = tuple(h * factor for h, factor in zip(grid.spacing, coarsening_factor))
    coarse_level = grid.level - 1
    return Grid(coarse_size, coarse_step_size, coarse_level)


def get_coarse_approximation(approximation: Approximation, coarsening_factor):
    return Approximation(f'{approximation.name}_c', get_coarse_grid(approximation.grid, coarsening_factor))


def get_coarse_rhs(rhs: RightHandSide, coarsening_factor):
    return RightHandSide(f'{rhs.name}_c', get_coarse_grid(rhs.grid, coarsening_factor))


def get_coarse_operator(operator, coarse_grid):
    return Operator(f'{operator.name}', coarse_grid, operator.stencil_generator)


class ConstantStencilGenerator:
    def __init__(self, stencil):
        self._stencil = stencil

    def generate_stencil(self, _):
        return self._stencil
