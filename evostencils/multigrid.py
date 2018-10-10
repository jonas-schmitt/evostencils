from evostencils.expressions import multigrid as mg
from evostencils.expressions import base
from evostencils.expressions import partitioning as part
from evostencils.types import matrix as matrix_types
from evostencils.types import grid as grid_types
from deap import gp


class Terminals:
    def __init__(self, operator, grid, dimension, coarsening_factor, interpolation_stencil_generator, restriction_stencil_generator):
        self.operator = operator
        self.grid = grid
        self.dimension = dimension
        self.coarsening_factor = coarsening_factor

        self.interpolation_stencil_generator = interpolation_stencil_generator
        self.restriction_stencil_generator = restriction_stencil_generator

        self.diagonal = base.Diagonal(operator)
        self.block_diagonal = base.BlockDiagonal(operator, tuple(2 for _ in range(self.dimension)))
        self.lower = base.LowerTriangle(operator)
        self.upper = base.UpperTriangle(operator)
        self.coarse_grid = mg.get_coarse_grid(self.grid, self.coarsening_factor)
        self.coarse_operator = mg.get_coarse_operator(self.operator, self.coarse_grid)
        self.interpolation = mg.get_interpolation(self.grid, self.coarse_grid, self.interpolation_stencil_generator)
        self.restriction = mg.get_restriction(self.grid, self.coarse_grid, self.restriction_stencil_generator)
        self.identity = base.Identity(self.operator.shape, self.grid)
        self.coarse_grid_solver = mg.CoarseGridSolver(self.coarse_operator)
        self.no_partitioning = part.Single
        self.red_black_partitioning = part.RedBlack


class Types:
    def __init__(self, terminals: Terminals):
        self.Operator = matrix_types.generate_matrix_type(terminals.operator.shape)
        self.LowerTriangularOperator = matrix_types.generate_lower_triangular_matrix_type(terminals.lower.shape)
        self.UpperTriangularOperator = matrix_types.generate_upper_triangular_matrix_type(terminals.lower.shape)
        self.Grid = grid_types.generate_grid_type(terminals.grid.size)
        self.Correction = grid_types.generate_correction_type(terminals.grid.size)
        self.RHS = grid_types.generate_rhs_type(terminals.grid.size)
        self.DiagonalOperator = matrix_types.generate_diagonal_matrix_type(terminals.diagonal.shape)
        self.BlockDiagonalOperator = matrix_types.generate_block_diagonal_matrix_type(terminals.block_diagonal.shape)
        self.Interpolation = matrix_types.generate_matrix_type(terminals.interpolation.shape)
        self.Restriction = matrix_types.generate_matrix_type(terminals.restriction.shape)
        self.CoarseOperator = matrix_types.generate_matrix_type(terminals.coarse_operator.shape)
        self.CoarseGrid = grid_types.generate_grid_type(terminals.coarse_grid.size)
        self.CoarseRHS = grid_types.generate_rhs_type(terminals.coarse_grid.size)
        self.Partitioning = part.Partitioning


def add_cycle(pset: gp.PrimitiveSetTyped, terminals: Terminals, level, types=None):
    if types is None:
        types = Types(terminals)
    pset.addTerminal(terminals.grid, types.Grid, f'u_{level}')
    null_grid = base.ZeroGrid(terminals.grid.size, terminals.grid.step_size)
    pset.addTerminal(null_grid, types.Correction, f'null_correction_{level}')
    #pset.addTerminal(base.ZeroGrid(terminals.coarse_grid.size, terminals.coarse_grid.step_size), types.CoarseRHS, f'zero_rhs_{level}')
    pset.addTerminal(terminals.operator, types.Operator, f'A_{level}')
    pset.addTerminal(terminals.identity, types.DiagonalOperator, f'I_{level}')
    pset.addTerminal(terminals.diagonal, types.DiagonalOperator, f'D_{level}')
    pset.addTerminal(terminals.lower, types.LowerTriangularOperator, f'L_{level}')
    pset.addTerminal(terminals.upper, types.UpperTriangularOperator, f'U_{level}')
    pset.addTerminal(terminals.block_diagonal, types.BlockDiagonalOperator, f'BD_{level}')
    #pset.addTerminal(terminals.coarse_grid_solver, types.CoarseOperator, f'S_{level}')
    #pset.addTerminal(terminals.interpolation, types.Interpolation, f'P_{level}')
    #pset.addTerminal(terminals.restriction, types.Restriction, f'R_{level}')
    pset.addTerminal(terminals.no_partitioning, types.Partitioning, f'no_{level}')
    pset.addTerminal(terminals.red_black_partitioning, types.Partitioning, f'rb_{level}')

    OperatorType = types.Operator
    GridType = types.Grid
    RHSType = types.RHS
    CorrectionType = types.Correction
    DiagonalOperatorType = types.DiagonalOperator
    BlockDiagonalOperatorType = types.BlockDiagonalOperator

    pset.addPrimitive(base.add, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, f'add_{level}')
    pset.addPrimitive(base.add, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'add_{level}')
    pset.addPrimitive(base.add, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'add_{level}')
    pset.addPrimitive(base.add, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, f'add_{level}')
    pset.addPrimitive(base.add, [OperatorType, OperatorType], OperatorType, f'add_{level}')

    pset.addPrimitive(base.sub, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, f'sub_{level}')
    pset.addPrimitive(base.sub, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'sub_{level}')
    pset.addPrimitive(base.sub, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'sub_{level}')
    pset.addPrimitive(base.sub, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, f'sub_{level}')
    pset.addPrimitive(base.sub, [OperatorType, OperatorType], OperatorType, f'sub_{level}')

    pset.addPrimitive(base.mul, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, f'mul_{level}')
    pset.addPrimitive(base.mul, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'mul_{level}')
    pset.addPrimitive(base.mul, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'mul_{level}')
    pset.addPrimitive(base.mul, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, f'mul_{level}')
    pset.addPrimitive(base.mul, [OperatorType, OperatorType], OperatorType, f'mul_{level}')

    pset.addPrimitive(base.minus, [OperatorType], OperatorType, f'minus_{level}')

    pset.addPrimitive(base.inv, [DiagonalOperatorType], DiagonalOperatorType, f'inverse_{level}')
    pset.addPrimitive(base.inv, [BlockDiagonalOperatorType], OperatorType, f'inverse_{level}')

    def create_cycle(grid, rhs, partitioning):
        def recursive_descent(expression: base.Expression):
            if isinstance(expression, base.Grid):
                if expression.grid.size == GridType.size:
                    return mg.cycle(grid, mg.residual(terminals.operator, grid, expression), partitioning)
                else:
                    raise RuntimeError("Something went terribly wrong")
            elif isinstance(expression, mg.Cycle):
                return mg.cycle(expression.iterate, recursive_descent(expression.correction), expression.partitioning, expression.weight)
            elif isinstance(expression, base.BinaryExpression):
                if expression.grid.size == GridType.size:
                    return mg.cycle(grid, expression, partitioning)
                else:
                    return type(expression)(expression.operand1, recursive_descent(expression.operand2))
            else:
                raise NotImplementedError("Not implemented")
        return recursive_descent(rhs)

    pset.addPrimitive(create_cycle, [GridType, RHSType, part.Partitioning], CorrectionType)

    def extend_cycle(root, operator):
        def recursive_descent(expression: base.Expression):
            if isinstance(expression, mg.Cycle):
                if expression.grid.size == GridType.size:
                    return mg.cycle(expression.iterate, base.mul(operator, expression.correction), expression.partitioning)
                else:
                    return mg.cycle(expression.iterate, recursive_descent(expression.correction), expression.partitioning, expression.weight)
            elif isinstance(expression, base.Grid):
                if expression.grid.size == GridType.size:
                    return base.mul(operator, expression)
                else:
                    raise RuntimeError("Something went terribly wrong")
            elif isinstance(expression, base.UnaryExpression):
                return type(expression)(recursive_descent(expression.operand))
            elif isinstance(expression, base.BinaryExpression):
                return type(expression)(expression.operand1, recursive_descent(expression.operand2))
            elif isinstance(expression, base.Scaling):
                return base.Scaling(expression.factor, recursive_descent(expression.operand))
            else:
                raise NotImplementedError("Not implemented")
        return recursive_descent(root)

    pset.addPrimitive(extend_cycle, [CorrectionType, OperatorType], CorrectionType, f"extend_cycle_{level}")

    def reduce_cycle(cycle):
        return cycle
    pset.addPrimitive(reduce_cycle, [CorrectionType], GridType, f"reduce_cycle_{level}")


    # Multigrid recipes
    CoarseGridType = types.CoarseGrid
    CoarseRHSType = types.CoarseRHS
    CoarseOperatorType = types.CoarseOperator
    InterpolationType = types.Interpolation
    RestrictionType = types.Restriction

    def noop(x):
        return x
    # Create intergrid operators
    #pset.addPrimitive(base.mul, [CoarseOperatorType, RestrictionType], RestrictionType, f'mul_{level}')
    #pset.addPrimitive(base.mul, [InterpolationType, CoarseOperatorType], InterpolationType, f'mul_{level}')
    #pset.addPrimitive(base.mul, [InterpolationType, RestrictionType], OperatorType, f'mul_{level}')

    #pset.addPrimitive(base.mul, [RestrictionType, types.Correction], CoarseRHSType, f'mul_{level}')
    #pset.addPrimitive(base.mul, [InterpolationType, CoarseRHSType], types.Correction, f'mul_{level}')
    #pset.addPrimitive(base.mul, [CoarseOperatorType, CoarseRHSType], CoarseRHSType, f'mul_{level}')

    #pset.addPrimitive(noop, [CoarseOperatorType], CoarseOperatorType, f'noop_{level}')
    pset.addPrimitive(noop, [part.Partitioning], part.Partitioning, f'noop_{level}')
    pset.addPrimitive(noop, [RHSType], RHSType, f'noop_{level}')


def generate_primitive_set(operator, grid, rhs, dimension, coarsening_factor,
                           interpolation_stencil=None, restriction_stencil=None, maximum_number_of_cycles=1):
    assert maximum_number_of_cycles >= 1, "The maximum number of cycles must be greater zero"
    terminals = Terminals(operator, grid, dimension, coarsening_factor, interpolation_stencil, restriction_stencil)
    types = Types(terminals)
    pset = gp.PrimitiveSetTyped("main", [], types.Grid)
    pset.addTerminal(rhs, types.RHS, 'f')
    add_cycle(pset, terminals, 0, types)
    for i in range(1, maximum_number_of_cycles):
        coarse_grid = base.ZeroGrid(terminals.coarse_grid.size, terminals.coarse_grid.step_size)
        terminals = Terminals(terminals.coarse_operator, coarse_grid, dimension, coarsening_factor,
                              interpolation_stencil, restriction_stencil)
        add_cycle(pset, terminals, base.ZeroGrid(terminals.coarse_grid.size, terminals.coarse_grid.step_size), i)

    return pset
