from evostencils.ir import base
from evostencils.ir import system
from evostencils.ir import partitioning as part
from evostencils.ir import smoother
from evostencils.ir.base import ConstantStencilGenerator
from evostencils.types import operator as operator_types
from evostencils.types import expression as expression_types
from evostencils.types import multiple
from evostencils.types import partitioning, level_control
from evostencils.types.wrapper import TypeWrapper
from evostencils.genetic_programming import PrimitiveSetTyped
from deap import gp
import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
import itertools
from functools import reduce


class NewtonSteps:
    pass


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
    equations_on_level = list(filter(lambda x: x.level == level, equations))
    system_operators = []
    for op_info in operators_on_level:
        if op_info.operator_type != base.Restriction and op_info.operator_type != base.Prolongation:
            system_operators.append(op_info)
    entries = []
    for equation in equations_on_level:
        row_of_entries = generate_operator_entries_from_equation(equation, system_operators, fields, grid)
        entries.append(row_of_entries)

    operator = system.Operator(f'A_{level}', entries)

    return operator


def generate_operators_from_l2_information(equations: [EquationInfo], operators: [OperatorInfo],
                                           fields: [sympy.Symbol], level, fine_grid: [base.Grid], coarse_grid: [base.Grid]):
    operators_on_level = list(filter(lambda x: x.level == level, operators))
    equations_on_level = list(filter(lambda x: x.level == level, equations))
    restriction_operators = []
    prolongation_operators = []
    system_operators = []
    for op_info in operators_on_level:
        if op_info.operator_type == base.Restriction:
            # TODO hacky solution for now
            if "gen_restrictionForSol" not in op_info.name:
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
    for equation in equations_on_level:
        row_of_entries = generate_operator_entries_from_equation(equation, system_operators, fields, fine_grid)
        entries.append(row_of_entries)

    operator = system.Operator(f'A_{level}', entries)

    return operator, restriction, prolongation


class Terminals:
    def __init__(self, approximation, operator, coarse_operator, restriction_operators, prolongation_operators, coarse_grid_solver, relaxation_factor_interval, partitionings=None):
        self.operator = operator
        self.coarse_operator = coarse_operator
        self.approximation = approximation
        self.prolongation_operators = prolongation_operators
        self.restriction_operators = restriction_operators
        self.coarse_grid_solver = coarse_grid_solver
        self.relaxation_factor_interval = relaxation_factor_interval
        self.no_partitioning = part.Single
        self.partitionings = partitionings

    @property
    def grid(self):
        return self.operator.grid

    @property
    def coarse_grid(self):
        return self.coarse_operator.grid


class Types:
    def __init__(self, terminals: Terminals, FinishedType, NotFinishedType, FAS=False):
        self.CoarseOperator = operator_types.generate_system_operator_type(terminals.coarse_operator.shape)
        types = [expression_types.generate_approximation_type(grid.size) for grid in terminals.grid]
        self.Approximation = multiple.generate_type_list(*types)
        types = [expression_types.generate_correction_type(grid.size) for grid in terminals.grid]
        self.Correction = multiple.generate_type_list(*types)
        types = [expression_types.generate_rhs_type(grid.size) for grid in terminals.grid]
        self.RHS = multiple.generate_type_list(*types)
        # Assumes prolongation and restriction operators to all have the same shape
        self.Prolongation = operator_types.generate_inter_grid_operator_type(terminals.prolongation_operators[0].shape)
        self.Restriction = operator_types.generate_inter_grid_operator_type(terminals.restriction_operators[0].shape)
        self.CoarseGridSolver = operator_types.generate_solver_type(terminals.coarse_operator.shape)
        types = [expression_types.generate_approximation_type(grid.size) for grid in terminals.coarse_grid]
        self.CoarseApproximation = multiple.generate_type_list(*types)
        types = [expression_types.generate_rhs_type(grid.size) for grid in terminals.coarse_grid]
        self.CoarseRHS = multiple.generate_type_list(*types)
        types = [expression_types.generate_correction_type(grid.size) for grid in terminals.coarse_grid]
        self.CoarseCorrection = multiple.generate_type_list(*types)
        self.Partitioning = partitioning.generate_any_partitioning_type()
        # self.BlockSize = TypeWrapper(typing.Tuple[typing.Tuple[int]])
        self.BlockSize = TypeWrapper(tuple)
        if FAS:
            self.NewtonSteps = TypeWrapper(NewtonSteps)
        self.Finished = FinishedType
        self.NotFinished = NotFinishedType


def add_level(pset: gp.PrimitiveSetTyped, terminals: Terminals, types: Types, level, relaxation_factor_samples=37,
              coarsest=False, FAS=False):
    if not coarsest:
        coarse_zero_approximation = system.ZeroApproximation(terminals.coarse_grid)
        pset.addTerminal(coarse_zero_approximation, types.CoarseApproximation, f'zero_{level + 1}')
        pset.addTerminal(terminals.coarse_operator, types.CoarseOperator, f'A_{level + 1}')
    for prolongation in terminals.prolongation_operators:
        pset.addTerminal(prolongation, types.Prolongation, prolongation.name)
    for restriction in terminals.restriction_operators:
        pset.addTerminal(restriction, types.Restriction, restriction.name)

    ApproximationType = types.Approximation
    scalar_equation = False
    if len(terminals.grid) == 1:
        scalar_equation = True
    CorrectionType = types.Correction

    def residual(state):
        approximation, rhs = state
        return base.Cycle(approximation, rhs, base.Residual(terminals.operator, approximation, rhs), predecessor=approximation.predecessor)

    def apply(operator, cycle):
        cycle.correction = base.Multiplication(operator, cycle.correction)
        return cycle

    def iterate(relaxation_factor_index, partitioning_, cycle):
        relaxation_factor = terminals.relaxation_factor_interval[relaxation_factor_index]
        rhs = cycle.rhs
        cycle.relaxation_factor = relaxation_factor
        cycle.partitioning = partitioning_
        approximation = cycle
        return approximation, rhs

    def initiate_cycle(coarse_operator, coarse_approximation, cycle):
        coarse_residual = base.Residual(coarse_operator, coarse_approximation, cycle.correction)
        new_cycle = base.Cycle(coarse_approximation, cycle.correction, coarse_residual)
        new_cycle.predecessor = cycle
        return new_cycle

    def coarse_grid_correction(prolongation_operator, state, restriction=None):
        cycle = state[0]
        if FAS:
            correction_FAS = base.mul(restriction, cycle.predecessor.approximation)  # Subract this term for FAS
            correction_c = base.sub(cycle, correction_FAS)
            correction = base.mul(prolongation_operator, correction_c)
        else:
            correction = base.Multiplication(prolongation_operator, cycle)
        cycle.predecessor.correction = correction
        return cycle.predecessor

    def restrict(restriction_operator, cycle):
        if FAS:
            # Special treatment for FAS
            residual_c = base.mul(restriction_operator, cycle.correction)
            residual_FAS = base.mul(terminals.coarse_operator, base.Multiplication(restriction_operator, cycle.approximation))  # Add this term for FAS
            residual_c = base.add(residual_c, residual_FAS)
            cycle.correction = residual_c
            return cycle
        else:
            return apply(restriction_operator, cycle)

    def coarsening(coarse_operator, coarse_approximation, restriction_operator, cycle):
        cycle = restrict(restriction_operator, cycle)
        return initiate_cycle(coarse_operator, coarse_approximation, cycle)

    def coarse_grid_correction_iteration(relaxation_factor_index, prolongation_operator, state, restriction_operator=None):
        cycle = coarse_grid_correction(prolongation_operator, state, restriction_operator)
        return iterate(relaxation_factor_index, terminals.no_partitioning, cycle)

    def smoothing_iteration(relaxation_factor_index, partitioning_, generate_smoother, cycle):
        assert isinstance(cycle.correction, base.Residual), 'Invalid production: expected residual'
        smoothing_operator = generate_smoother(cycle.correction.operator)
        cycle = apply(base.Inverse(smoothing_operator), cycle)
        return iterate(relaxation_factor_index, partitioning_, cycle)

    def decoupled_jacobi(relaxation_factor_index, partitioning_, cycle):
        return smoothing_iteration(relaxation_factor_index, partitioning_, smoother.generate_decoupled_jacobi, cycle)

    def collective_jacobi(relaxation_factor_index, partitioning_, cycle):
        return smoothing_iteration(relaxation_factor_index, partitioning_, smoother.generate_collective_jacobi, cycle)

    def collective_block_jacobi(relaxation_factor_index, block_size, cycle):
        def generate_collective_block_jacobi_fixed(operator):
            return smoother.generate_collective_block_jacobi(operator, block_size)

        return smoothing_iteration(relaxation_factor_index, part.Single, generate_collective_block_jacobi_fixed, cycle)

    def jacobi_picard(relaxation_factor_index, partitioning_, cycle):
        return smoothing_iteration(relaxation_factor_index, partitioning_, smoother.generate_jacobi_picard, cycle)

    def jacobi_newton(relaxation_factor_index, partitioning_, n_newton_steps, cycle):
        def generate_jacobi_newton_fixed(operator):
            return smoother.generate_jacobi_newton(operator, n_newton_steps)

        return smoothing_iteration(relaxation_factor_index, partitioning_, generate_jacobi_newton_fixed, cycle)

    pset.addPrimitive(residual, [multiple.generate_type_list(types.Approximation, types.RHS, types.Finished)], multiple.generate_type_list(ApproximationType, CorrectionType, types.Finished), f"residual_{level}")
    pset.addPrimitive(residual, [multiple.generate_type_list(types.Approximation, types.RHS, types.NotFinished)], multiple.generate_type_list(ApproximationType, CorrectionType, types.NotFinished), f"residual_{level}")

    if not scalar_equation:
        pset.addPrimitive(decoupled_jacobi, [TypeWrapper(int), types.Partitioning, multiple.generate_type_list(types.Approximation, types.Correction, types.Finished)],
                          multiple.generate_type_list(types.Approximation, types.RHS, types.Finished), f"decoupled_jacobi_{level}")
        pset.addPrimitive(decoupled_jacobi, [TypeWrapper(int), types.Partitioning, multiple.generate_type_list(types.Approximation, types.Correction, types.NotFinished)],
                          multiple.generate_type_list(types.Approximation, types.RHS, types.NotFinished), f"decoupled_jacobi_{level}")

    # start: Exclude for FAS
    if not FAS:
        pset.addPrimitive(collective_jacobi, [TypeWrapper(int), types.Partitioning, multiple.generate_type_list(types.Approximation, types.Correction, types.Finished)],
                          multiple.generate_type_list(types.Approximation, types.RHS, types.Finished), f"collective_jacobi_{level}")
        pset.addPrimitive(collective_jacobi, [TypeWrapper(int), types.Partitioning, multiple.generate_type_list(types.Approximation, types.Correction, types.NotFinished)],
                          multiple.generate_type_list(types.Approximation, types.RHS, types.NotFinished), f"collective_jacobi_{level}")
        pset.addPrimitive(collective_block_jacobi, [TypeWrapper(int), types.BlockSize, multiple.generate_type_list(types.Approximation, types.Correction, types.Finished)],
                          multiple.generate_type_list(types.Approximation, types.RHS, types.Finished), f"collective_block_jacobi_{level}")
        pset.addPrimitive(collective_block_jacobi, [TypeWrapper(int), types.BlockSize, multiple.generate_type_list(types.Approximation, types.Correction, types.NotFinished)],
                          multiple.generate_type_list(types.Approximation, types.RHS, types.NotFinished), f"collective_block_jacobi_{level}")
    # end : Exclude for FAS
    if FAS:
        pset.addPrimitive(jacobi_picard, [TypeWrapper(int), types.Partitioning, multiple.generate_type_list(types.Approximation, types.Correction, types.Finished)],
                          multiple.generate_type_list(types.Approximation, types.RHS, types.Finished), f"jacobi_picard_{level}")
        pset.addPrimitive(jacobi_picard, [TypeWrapper(int), types.Partitioning, multiple.generate_type_list(types.Approximation, types.Correction, types.NotFinished)],
                          multiple.generate_type_list(types.Approximation, types.RHS, types.NotFinished), f"jacobi_picard_{level}")
        pset.addPrimitive(jacobi_newton, [TypeWrapper(int), types.Partitioning, types.NewtonSteps, multiple.generate_type_list(types.Approximation, types.Correction, types.Finished)],
                          multiple.generate_type_list(types.Approximation, types.RHS, types.Finished), f"jacobi_newton_{level}")
        pset.addPrimitive(jacobi_newton, [TypeWrapper(int), types.Partitioning, types.NewtonSteps, multiple.generate_type_list(types.Approximation, types.Correction, types.NotFinished)],
                          multiple.generate_type_list(types.Approximation, types.RHS, types.NotFinished), f"jacobi_newton_{level}")

    if not coarsest:
        if FAS:
            pset.addPrimitive(coarse_grid_correction_iteration, [TypeWrapper(int), types.Prolongation, multiple.generate_type_list(types.CoarseApproximation, types.CoarseRHS, types.Finished), types.Restriction],
                              multiple.generate_type_list(types.Approximation, types.RHS, types.Finished), f"iterate_cgc_{level}")
            if level > 0:
                pset.addPrimitive(coarse_grid_correction_iteration, [TypeWrapper(int), types.Prolongation, multiple.generate_type_list(types.CoarseApproximation, types.CoarseRHS, types.NotFinished), types.Restriction],
                                  multiple.generate_type_list(types.Approximation, types.RHS, types.NotFinished), f"iterate_cgc_{level}")
        else:
            pset.addPrimitive(coarse_grid_correction_iteration, [TypeWrapper(int), types.Prolongation, multiple.generate_type_list(types.CoarseApproximation, types.CoarseRHS, types.Finished)],
                              multiple.generate_type_list(types.Approximation, types.RHS, types.Finished), f"iterate_cgc_{level}")
            if level > 0:
                pset.addPrimitive(coarse_grid_correction_iteration, [TypeWrapper(int), types.Prolongation, multiple.generate_type_list(types.CoarseApproximation, types.CoarseRHS, types.NotFinished)],
                          multiple.generate_type_list(types.Approximation, types.RHS, types.NotFinished), f"iterate_cgc_{level}")

        pset.addPrimitive(coarsening,
                          [types.CoarseOperator, types.CoarseApproximation, types.Restriction, multiple.generate_type_list(types.Approximation, types.Correction, types.NotFinished)],
                          multiple.generate_type_list(types.CoarseApproximation, types.CoarseCorrection, types.NotFinished),
                          f"coarsening_{level}")
        pset.addPrimitive(coarsening,
                          [types.CoarseOperator, types.CoarseApproximation, types.Restriction, multiple.generate_type_list(types.Approximation, types.Correction, types.Finished)],
                          multiple.generate_type_list(types.CoarseApproximation, types.CoarseCorrection, types.Finished),
                          f"coarsening_{level}")

    else:
        def exact_coarse_grid_correction_iteration(relaxation_factor_index, prolongation_operator, coarse_grid_solver, cycle, restriction_operator=None):
            if FAS:
                approximation_c = base.mul(coarse_grid_solver, cycle.correction)
                restricted_solution_FAS = base.mul(restriction_operator, cycle.approximation)
                correction = base.mul(prolongation_operator, base.sub(approximation_c, restricted_solution_FAS))  # Subtract term for FAS
                cycle.correction = correction
            else:
                cycle = apply(prolongation_operator, apply(coarse_grid_solver, cycle))
            return iterate(relaxation_factor_index, terminals.no_partitioning, cycle)

        def restrict_solve_prolongate(relaxation_factor_index, prolongation_operator, coarse_grid_solver, restriction_operator, cycle):
            cycle = restrict(restriction_operator, cycle)
            return exact_coarse_grid_correction_iteration(relaxation_factor_index, prolongation_operator, coarse_grid_solver, cycle, restriction_operator)

        pset.addPrimitive(restrict_solve_prolongate, [TypeWrapper(int), types.Prolongation, types.CoarseGridSolver, types.Restriction, multiple.generate_type_list(types.Approximation, types.Correction, types.NotFinished)],
                          multiple.generate_type_list(types.Approximation, types.RHS, types.Finished),
                          f'prolongate_solve_restrict_{level}')
        pset.addPrimitive(restrict_solve_prolongate, [TypeWrapper(int), types.Prolongation, types.CoarseGridSolver, types.Restriction, multiple.generate_type_list(types.Approximation, types.Correction, types.Finished)],
                          multiple.generate_type_list(types.Approximation, types.RHS, types.Finished),
                          f'restrict_solve_prolongate{level}')

        pset.addTerminal(terminals.coarse_grid_solver, types.CoarseGridSolver, f'CGS_{level}')

def generate_primitive_set(approximation, rhs, dimension, coarsening_factors, max_level, equations, operators, fields,
                           maximum_local_system_size=8, relaxation_factor_samples=37,
                           coarse_grid_solver_expression=None, depth=2, enable_partitioning=True, LevelFinishedType=None, LevelNotFinishedType=None,
                           FAS=False):
    assert depth >= 1, "The maximum number of levels must be greater zero"
    coarsest = False
    if depth == 1:
        coarsest = True
    fine_grid = approximation.grid
    coarse_grid = system.get_coarse_grid(fine_grid, coarsening_factors)
    operator, restriction, prolongation, = \
        generate_operators_from_l2_information(equations, operators, fields, max_level, fine_grid, coarse_grid)
    coarse_operator, coarse_restriction, coarse_prolongation, = \
        generate_operators_from_l2_information(equations, operators, fields, max_level - 1, coarse_grid, system.get_coarse_grid(coarse_grid, coarsening_factors))
    # For now assumes that only one prolongation, restriction and partitioning operator is available
    # TODO: Extend in the future
    partitionings = [part.RedBlack]
    restriction_operators = [restriction]
    prolongation_operators = [prolongation]
    coarse_grid_solver = base.CoarseGridSolver("Coarse-Grid Solver", coarse_operator, coarse_grid_solver_expression)
    relaxation_factor_interval = np.linspace(0.1, 1.9, relaxation_factor_samples)
    terminals = Terminals(approximation, operator, coarse_operator, restriction_operators, prolongation_operators, coarse_grid_solver, relaxation_factor_interval, partitionings)
    if LevelFinishedType is None:
        LevelFinishedType = level_control.generate_finished_type()
    if LevelNotFinishedType is None:
        LevelNotFinishedType = level_control.generate_not_finished_type()
    types = Types(terminals, LevelFinishedType, LevelNotFinishedType, FAS=FAS)
    pset = PrimitiveSetTyped("main", [], multiple.generate_type_list(types.Approximation, types.RHS, types.Finished))
    pset.addTerminal((approximation, rhs), multiple.generate_type_list(types.Approximation, types.RHS, types.NotFinished), 'u_and_f')
    pset.addTerminal(terminals.no_partitioning, types.Partitioning, terminals.no_partitioning.get_name())
    # Start: Exclude for FAS
    if enable_partitioning:
        for p in terminals.partitionings:
            pset.addTerminal(p, types.Partitioning, p.get_name())
    # End: Exclude for FAS
    for i in range(0, relaxation_factor_samples):
        pset.addTerminal(i, TypeWrapper(int))

    # Block sizes
    # Start: not need for FAS
    if not FAS:
        block_sizes = []
        for i in range(len(fields)):
            block_sizes.append([])

            def generate_block_size(block_size_, block_size_max, dimension_):
                if dimension_ == 1:
                    for k in range(1, block_size_max + 1):
                        block_sizes[-1].append(block_size_ + (k,))
                else:
                    for k in range(1, block_size_max + 1):
                        generate_block_size(block_size_ + (k,), block_size_max, dimension_ - 1)

            generate_block_size((), maximum_local_system_size, dimension)
        for block_size_permutation in itertools.product(*block_sizes):
            number_of_terms = 0
            for block_size in block_size_permutation:
                number_of_terms += reduce(lambda x, y: x * y, block_size)
            if len(approximation.grid) < number_of_terms <= maximum_local_system_size:
                pset.addTerminal(block_size_permutation, types.BlockSize)
    # End: not need for FAS
    # Newton Steps
    if FAS:
        newton_steps = [1, 2, 3, 4]
        for i in newton_steps:
            pset.addTerminal(i, types.NewtonSteps)

    add_level(pset, terminals, types, 0, relaxation_factor_samples, coarsest=coarsest, FAS=FAS)

    terminal_list = [terminals]
    for i in range(1, depth):
        approximation = system.ZeroApproximation(terminals.coarse_grid)
        operator = coarse_operator
        prolongation_operators = [coarse_prolongation]
        restriction_operators = [coarse_restriction]
        fine_grid = terminals.coarse_grid
        coarse_grid = system.get_coarse_grid(fine_grid, coarsening_factors)
        coarsest = False
        if i == depth - 1:
            coarsest = True
            coarse_operator = \
                generate_system_operator_from_l2_information(equations, operators, fields, max_level - i - 1,
                                                             coarse_grid)
        else:
            coarse_operator, coarse_restriction, coarse_prolongation = \
                generate_operators_from_l2_information(equations, operators, fields, max_level - i - 1, coarse_grid,
                                                       system.get_coarse_grid(coarse_grid, coarsening_factors))

        coarse_grid_solver = base.CoarseGridSolver("Coarse-Grid Solver", coarse_operator, coarse_grid_solver_expression)
        terminals = Terminals(approximation, operator, coarse_operator, restriction_operators, prolongation_operators, coarse_grid_solver, relaxation_factor_interval, partitionings)
        types = Types(terminals, LevelFinishedType, LevelNotFinishedType, FAS=FAS)
        add_level(pset, terminals, types, i, relaxation_factor_samples, coarsest=coarsest, FAS=FAS)
        terminal_list.append(terminals)

    return pset, terminal_list
