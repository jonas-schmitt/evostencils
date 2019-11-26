from evostencils.expressions import base
from evostencils.expressions import system
from evostencils.expressions import partitioning as part
from evostencils.expressions import smoother
from evostencils.expressions.base import ConstantStencilGenerator
from evostencils.types import operator as matrix_types
from evostencils.types import grid as grid_types
from evostencils.types import multiple
from evostencils.types import partitioning, level_control
from evostencils.types.wrapper import TypeWrapper
from evostencils.genetic_programming import PrimitiveSetTyped
from deap import gp
import sympy
from sympy.parsing.sympy_parser import parse_expr
import itertools
import typing
from functools import reduce
import random


class OperatorInfo:
    def __init__(self, name, level, stencil, operator_type=base.Operator):
        self._name = name
        self._level = level
        self._stencil = stencil
        self._associated_field = None
        self._operator_type = operator_type

    @property
    def name(self):
        return self._name

    @property
    def level(self):
        return self._level

    @property
    def stencil(self):
        return self._stencil

    @property
    def operator_type(self):
        return self._operator_type


class EquationInfo:
    def __init__(self, name: str, level: int, expr_str: str):
        self._name = name
        self._level = level
        transformed_expr = ''
        tokens = expr_str.split(' ')
        for token in tokens:
            transformed_expr += ' ' + token.split('@')[0]
        tmp = transformed_expr.split('==')
        self._sympy_expr = parse_expr(tmp[0])
        self._rhs_name = tmp[1].strip(' ')
        self._associated_field = None

    @property
    def name(self):
        return self._name

    @property
    def level(self):
        return self._level

    @property
    def sympy_expr(self):
        return self._sympy_expr

    @property
    def rhs_name(self):
        return self._rhs_name

    @property
    def associated_field(self):
        return self._associated_field


def generate_operator_entries_from_equation(equation, operators: list, fields, grid):
    row_of_operators = []
    indices = []

    def recursive_descent(expr, field_index):
        if expr.is_Number:
            identity = base.Identity(grid[field_index])
            if not expr == sympy.sympify(1):
                return base.Scaling(float(expr.evalf()), identity)
            else:
                return identity
        elif expr.is_Symbol:
            op_symbol = expr
            j = next(k for k, op_info in enumerate(operators) if op_symbol.name == op_info.name)
            operator = base.Operator(op_symbol.name, grid[field_index], ConstantStencilGenerator(operators[j].stencil))
            return operator
        elif expr.is_Mul:
            tmp = recursive_descent(expr.args[-1], field_index)
            for arg in expr.args[-2::-1]:
                if arg.is_Number:
                    tmp = base.Scaling(float(arg.evalf()), tmp)
                else:
                    lhs = recursive_descent(arg, field_index)
                    tmp = base.Multiplication(lhs, tmp)
        elif expr.is_Add:
            tmp = recursive_descent(expr.args[0], field_index)
            for arg in expr.args[1:]:
                tmp = base.Addition(recursive_descent(arg, field_index), tmp)
        else:
            raise RuntimeError("Invalid Expression")
        return tmp

    expanded_expression = sympy.expand(equation.sympy_expr)
    for i, field in enumerate(fields):
        if field in expanded_expression.free_symbols:
            collected_terms = sympy.collect(expanded_expression, field, evaluate=False)
            term = collected_terms[field]
            entry = recursive_descent(term, i)
            row_of_operators.append(entry)
            indices.append(i)
    for i in range(len(grid)):
        if i not in indices:
            row_of_operators.append(base.ZeroOperator(grid[i]))
            indices.append(i)
    result = [operator for (index, operator) in sorted(zip(indices, row_of_operators), key=lambda p: p[0])]
    return result


def generate_system_operator_from_l2_information(equations: [EquationInfo], operators: [OperatorInfo],
                                                 fields: [sympy.Symbol], level, grid: [base.Grid]):
    operators_on_level = list(filter(lambda x: x.level == level, operators))
    system_operators = []
    for op_info in operators_on_level:
        if op_info.operator_type != base.Restriction and op_info.operator_type != base.Prolongation:
            system_operators.append(op_info)
    entries = []
    for equation in equations:
        row_of_entries = generate_operator_entries_from_equation(equation, system_operators, fields, grid)
        entries.append(row_of_entries)

    operator = system.Operator(f'A_{level}', entries)

    return operator


def generate_operators_from_l2_information(equations: [EquationInfo], operators: [OperatorInfo],
                                           fields: [sympy.Symbol], level, fine_grid: [base.Grid], coarse_grid: [base.Grid]):
    operators_on_level = list(filter(lambda x: x.level == level, operators))
    restriction_operators = []
    prolongation_operators = []
    system_operators = []
    for op_info in operators_on_level:
        if op_info.operator_type == base.Restriction:
            restriction_operators.append(op_info)
        elif op_info.operator_type == base.Prolongation:
            prolongation_operators.append(op_info)
        else:
            system_operators.append(op_info)
    assert len(restriction_operators) == len(fields), 'The number of restriction operators does not match with the number of fields'
    assert len(prolongation_operators) == len(fields), 'The number of prolongation operators does not match with the number of fields'
    list_of_restriction_operators = [base.Restriction(op_info.name, fine_grid[i], coarse_grid[i], ConstantStencilGenerator(op_info.stencil))
                                     for i, op_info in enumerate(restriction_operators)]
    restriction = system.Restriction(f'R_{level}', list_of_restriction_operators)

    list_of_prolongation_operators = [base.Prolongation(op_info.name, fine_grid[i], coarse_grid[i], ConstantStencilGenerator(op_info.stencil))
                                      for i, op_info in enumerate(prolongation_operators)]
    prolongation = system.Prolongation(f'P_{level}', list_of_prolongation_operators)

    entries = []
    for equation in equations:
        row_of_entries = generate_operator_entries_from_equation(equation, system_operators, fields, fine_grid)
        entries.append(row_of_entries)

    operator = system.Operator(f'A_{level}', entries)

    return operator, restriction, prolongation


class Terminals:
    def __init__(self, approximation, dimension, coarsening_factors, operator, coarse_operator, restriction, prolongation,
                 coarse_grid_solver_expression=None):
        self.operator = operator
        self.approximation = approximation
        self.grid = approximation.grid
        self.dimension = dimension
        self.coarsening_factor = coarsening_factors
        self.prolongation = prolongation
        self.restriction = restriction
        self.coarse_grid = system.get_coarse_grid(self.approximation.grid, self.coarsening_factor)
        self.coarse_operator = coarse_operator
        self.identity = system.Identity(approximation.grid)
        self.coarse_grid_solver = base.CoarseGridSolver(self.coarse_operator, expression=coarse_grid_solver_expression)
        self.no_partitioning = part.Single
        self.red_black_partitioning = part.RedBlack


class Types:
    def __init__(self, terminals: Terminals, FinishedType, NotFinishedType):
        types = [grid_types.generate_grid_type(grid.size) for grid in terminals.grid]
        self.Grid = multiple.generate_type_list(*types)
        types = [grid_types.generate_correction_type(grid.size) for grid in terminals.grid]
        self.Correction = multiple.generate_type_list(*types)
        types = [grid_types.generate_rhs_type(grid.size) for grid in terminals.grid]
        self.RHS = multiple.generate_type_list(*types)
        self.Prolongation = matrix_types.generate_inter_grid_operator_type(terminals.prolongation.shape)
        self.Restriction = matrix_types.generate_inter_grid_operator_type(terminals.restriction.shape)
        self.CoarseGridSolver = matrix_types.generate_solver_type(terminals.coarse_operator.shape)
        types = [grid_types.generate_grid_type(grid.size) for grid in terminals.coarse_grid]
        self.CoarseGrid = multiple.generate_type_list(*types)
        types = [grid_types.generate_rhs_type(grid.size) for grid in terminals.coarse_grid]
        self.CoarseRHS = multiple.generate_type_list(*types)
        types = [grid_types.generate_correction_type(grid.size) for grid in terminals.coarse_grid]
        self.CoarseCorrection = multiple.generate_type_list(*types)
        self.Partitioning = partitioning.generate_any_partitioning_type()
        self.BlockSize = TypeWrapper(typing.Tuple[typing.Tuple[int]])
        self.Finished = FinishedType
        self.NotFinished = NotFinishedType


def add_cycle(pset: gp.PrimitiveSetTyped, terminals: Terminals, types: Types, level, coarsest=False):
    null_grid_coarse = system.ZeroApproximation(terminals.coarse_grid)
    pset.addTerminal(null_grid_coarse, types.CoarseGrid, f'zero_grid_{level+1}')
    pset.addTerminal(terminals.prolongation, types.Prolongation, f'P_{level}')
    pset.addTerminal(terminals.restriction, types.Restriction, f'R_{level}')

    GridType = types.Grid
    CorrectionType = types.Correction

    def coarse_cycle(coarse_grid, cycle):
        result = base.Cycle(cycle.approximation, cycle.rhs,
                            base.Cycle(coarse_grid, cycle.correction,
                                       base.Residual(terminals.coarse_operator, coarse_grid, cycle.correction)),
                            predecessor=cycle.predecessor)
        result.correction.predecessor = result
        return result.correction

    def residual(args):
        return base.Cycle(args[0], args[1], base.Residual(terminals.operator, args[0], args[1]), predecessor=args[0].predecessor)

    def restrict(operator, cycle):
        return base.Cycle(cycle.approximation, cycle.rhs, base.mul(operator, cycle.correction), predecessor=cycle.predecessor)

    def coarse_grid_correction(interpolation, args, relaxation_factor):
        cycle = args[0]
        cycle.predecessor._correction = base.mul(interpolation, cycle)
        return iterate(cycle.predecessor, relaxation_factor)

    def iterate(cycle, relaxation_factor):
        rhs = cycle.rhs
        approximation = base.Cycle(cycle.approximation, cycle.rhs, cycle.correction, cycle.partitioning,
                                   relaxation_factor, cycle.predecessor)
        return approximation, rhs

    def smoothing(generate_smoother, cycle, partitioning, relaxation_factor):
        assert isinstance(cycle.correction, base.Residual), 'Invalid production'
        approximation = cycle.approximation
        rhs = cycle.rhs
        smoothing_operator = generate_smoother(cycle.correction.operator)
        correction = base.Multiplication(base.Inverse(smoothing_operator), cycle.correction)
        return iterate(base.Cycle(approximation, rhs, correction, partitioning=partitioning,
                                  predecessor=cycle.predecessor), relaxation_factor)

    def decoupled_jacobi(cycle, partitioning, relaxation_factor, _):
        return smoothing(smoother.generate_decoupled_jacobi, cycle, partitioning, relaxation_factor)

    def collective_jacobi(cycle, partitioning, relaxation_factor, _):
        return smoothing(smoother.generate_collective_jacobi, cycle, partitioning, relaxation_factor)

    def collective_block_jacobi(cycle, partitioning, relaxation_factor, block_size):
        def generate_collective_block_jacobi_fixed(operator):
            return smoother.generate_collective_block_jacobi(operator, block_size)
        return smoothing(generate_collective_block_jacobi_fixed, cycle, partitioning, relaxation_factor)

    pset.addPrimitive(iterate, [multiple.generate_type_list(types.Grid, types.Correction, types.Finished), TypeWrapper(float)], multiple.generate_type_list(types.Grid, types.RHS, types.Finished), f"iterate_{level}")
    pset.addPrimitive(iterate, [multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished), TypeWrapper(float)], multiple.generate_type_list(types.Grid, types.RHS, types.NotFinished), f"iterate_{level}")

    pset.addPrimitive(residual, [multiple.generate_type_list(types.Grid, types.RHS, types.Finished)], multiple.generate_type_list(GridType, CorrectionType, types.Finished), f"residual_{level}")
    pset.addPrimitive(residual, [multiple.generate_type_list(types.Grid, types.RHS, types.NotFinished)], multiple.generate_type_list(GridType, CorrectionType, types.NotFinished), f"residual_{level}")

    pset.addPrimitive(decoupled_jacobi, [multiple.generate_type_list(types.Grid, types.Correction, types.Finished), types.Partitioning, TypeWrapper(float), types.BlockSize], multiple.generate_type_list(types.Grid, types.RHS, types.Finished), f"decoupled_jacobi_{level}")
    pset.addPrimitive(decoupled_jacobi, [multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished), types.Partitioning, TypeWrapper(float), types.BlockSize], multiple.generate_type_list(types.Grid, types.RHS, types.NotFinished), f"decoupled_jacobi_{level}")

    pset.addPrimitive(collective_jacobi, [multiple.generate_type_list(types.Grid, types.Correction, types.Finished), types.Partitioning, TypeWrapper(float), types.BlockSize], multiple.generate_type_list(types.Grid, types.RHS, types.Finished), f"collective_jacobi_{level}")
    pset.addPrimitive(collective_jacobi, [multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished), types.Partitioning, TypeWrapper(float), types.BlockSize], multiple.generate_type_list(types.Grid, types.RHS, types.NotFinished), f"collective_jacobi_{level}")

    pset.addPrimitive(collective_block_jacobi, [multiple.generate_type_list(types.Grid, types.Correction, types.Finished), types.Partitioning, TypeWrapper(float), types.BlockSize], multiple.generate_type_list(types.Grid, types.RHS, types.Finished), f"collective_block_jacobi_{level}")
    pset.addPrimitive(collective_block_jacobi, [multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished), types.Partitioning, TypeWrapper(float), types.BlockSize], multiple.generate_type_list(types.Grid, types.RHS, types.NotFinished), f"collective_block_jacobi_{level}")

    if not coarsest:

        pset.addPrimitive(coarse_grid_correction, [types.Prolongation, multiple.generate_type_list(types.CoarseGrid, types.CoarseRHS, types.Finished), TypeWrapper(float)], multiple.generate_type_list(types.Grid, types.RHS, types.Finished), f"cgc_{level}")

        pset.addPrimitive(coarse_cycle,
                          [types.CoarseGrid, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.NotFinished)],
                          multiple.generate_type_list(types.CoarseGrid, types.CoarseCorrection, types.NotFinished),
                          f"coarse_cycle_{level}")
        pset.addPrimitive(coarse_cycle,
                          [types.CoarseGrid, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished)],
                          multiple.generate_type_list(types.CoarseGrid, types.CoarseCorrection, types.Finished),
                          f"coarse_cycle_{level}")

    else:
        def solve(cgs, interpolation, cycle):
            new_cycle = base.Cycle(cycle.approximation, cycle.rhs, base.mul(interpolation, base.mul(cgs, cycle.correction)),
                                   predecessor=cycle.predecessor)
            return iterate(new_cycle, 1)

        pset.addPrimitive(solve, [types.CoarseGridSolver, types.Prolongation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.NotFinished)],
                          multiple.generate_type_list(types.Grid, types.RHS, types.Finished),
                          f'solve_{level}')
        pset.addPrimitive(solve, [types.CoarseGridSolver, types.Prolongation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished)],
                          multiple.generate_type_list(types.Grid, types.RHS, types.Finished),
                          f'solve_{level}')

        pset.addTerminal(terminals.coarse_grid_solver, types.CoarseGridSolver, f'CGS_{level}')

    # Multigrid recipes
    pset.addPrimitive(restrict, [types.Restriction, multiple.generate_type_list(types.Grid, types.Correction, types.Finished)],
                      multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished),
                      f'restrict_{level}')
    pset.addPrimitive(restrict, [types.Restriction, multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished)],
                      multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.NotFinished),
                      f'restrict_{level}')


def generate_primitive_set(approximation, rhs, dimension, coarsening_factors, max_level, equations, operators, fields, maximum_block_size=2,
                           coarse_grid_solver_expression=None, depth=2, LevelFinishedType=None, LevelNotFinishedType=None):
    assert depth >= 1, "The maximum number of cycles must be greater zero"
    coarsest = False
    cgs_expression = None
    if depth == 1:
        coarsest = True
        cgs_expression = coarse_grid_solver_expression
    fine_grid = approximation.grid
    coarse_grid = system.get_coarse_grid(fine_grid, coarsening_factors)
    operator, restriction, prolongation, = \
        generate_operators_from_l2_information(equations, operators, fields, max_level, fine_grid, coarse_grid)
    coarse_operator, coarse_restriction, coarse_prolongation, = \
        generate_operators_from_l2_information(equations, operators, fields, max_level-1, coarse_grid, system.get_coarse_grid(coarse_grid, coarsening_factors))
    terminals = Terminals(approximation, dimension, coarsening_factors, operator, coarse_operator, restriction, prolongation, cgs_expression)
    if LevelFinishedType is None:
        LevelFinishedType = level_control.generate_finished_type()
    if LevelNotFinishedType is None:
        LevelNotFinishedType = level_control.generate_not_finished_type()
    types = Types(terminals, LevelFinishedType, LevelNotFinishedType)
    pset = PrimitiveSetTyped("main", [], multiple.generate_type_list(types.Grid, types.RHS, types.Finished))
    pset.addTerminal((approximation, rhs), multiple.generate_type_list(types.Grid, types.RHS, types.NotFinished), 'u_and_f')
    pset.addTerminal(terminals.no_partitioning, types.Partitioning, f'no')
    pset.addTerminal(terminals.red_black_partitioning, types.Partitioning, f'red_black')

    pset.addTerminal(1.0, TypeWrapper(float), f'omega')

    def sample_relaxation_factor():
        return random.gauss(1.0, 0.4)
    for i in range(1000):
        pset.addTerminal(sample_relaxation_factor(), TypeWrapper(float), f'omega_{i}')
    block_sizes = []
    for i in range(len(fields)):
        block_sizes.append([])

        def generate_block_size(block_size, block_size_max, dimension):
            if dimension == 1:
                for k in range(1, block_size_max + 1):
                    block_sizes[-1].append(block_size + (k,))
            else:
                for k in range(1, block_size_max + 1):
                    generate_block_size(block_size + (k,), block_size_max, dimension - 1)
        generate_block_size((), maximum_block_size, dimension)
    maximum_number_of_generatable_terms = 7
    for block_size_permutation in itertools.product(*block_sizes):
        number_of_terms = 0
        for block_size in block_size_permutation:
            number_of_terms += reduce(lambda x, y: x * y, block_size)
        if number_of_terms <= maximum_number_of_generatable_terms:
            pset.addTerminal(block_size_permutation, types.BlockSize)

    add_cycle(pset, terminals, types, 0, coarsest)
    for i in range(1, depth):
        approximation = system.ZeroApproximation(terminals.coarse_grid)
        operator = coarse_operator
        prolongation = coarse_prolongation
        restriction = coarse_restriction
        fine_grid = terminals.coarse_grid
        coarse_grid = system.get_coarse_grid(fine_grid, coarsening_factors)
        cgs_expression = None
        coarsest = False
        if i == depth - 1:
            coarsest = True
            cgs_expression = coarse_grid_solver_expression
            coarse_operator = \
                generate_system_operator_from_l2_information(equations, operators, fields, max_level - i - 1,
                                                             coarse_grid)
        else:
            coarse_operator, coarse_restriction, coarse_prolongation = \
                generate_operators_from_l2_information(equations, operators, fields, max_level - i - 1, coarse_grid,
                                                       system.get_coarse_grid(coarse_grid, coarsening_factors))

        terminals = Terminals(approximation, dimension, coarsening_factors, operator, coarse_operator, restriction,
                              prolongation, cgs_expression)
        types = Types(terminals, LevelFinishedType, LevelNotFinishedType)
        add_cycle(pset, terminals, types, i, coarsest)

    return pset