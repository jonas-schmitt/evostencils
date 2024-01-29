from evostencils.ir import base
from evostencils.ir import system
from evostencils.ir import partitioning as part
from evostencils.ir import smoother
from evostencils.ir.base import ConstantStencilGenerator
from evostencils.grammar.typing import Type
from evostencils.grammar.gp import PrimitiveSetTyped
import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
import itertools
from functools import reduce

use_hypre = False
use_hyteg = True
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
                                                 fields: [sympy.Symbol], level, depth, grid: [base.Grid]):
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

    operator = system.Operator(f'A_{depth}', entries)

    return operator


def generate_operators_from_l2_information(equations: [EquationInfo], operators: [OperatorInfo],
                                           fields: [sympy.Symbol], level, depth, fine_grid: [base.Grid], coarse_grid: [base.Grid]):
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
    restriction = system.Restriction(f'R_{depth}', list_of_restriction_operators)

    list_of_prolongation_operators = [base.Prolongation(op_info.name, fine_grid[i], coarse_grid[i], ConstantStencilGenerator(op_info.stencil))
                                      for i, op_info in enumerate(prolongation_operators)]
    prolongation = system.Prolongation(f'P_{depth+1}', list_of_prolongation_operators)

    entries = []
    for equation in equations_on_level:
        row_of_entries = generate_operator_entries_from_equation(equation, system_operators, fields, fine_grid)
        entries.append(row_of_entries)

    operator = system.Operator(f'A_{depth}', entries)

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
    @staticmethod
    def _init_type(identifier, types, type_attribute=None, guard=False):
        if type_attribute is None:
            type_attribute = identifier
        if types is None:
            return Type(identifier, guard)
        else:
            return getattr(types, type_attribute)

    def __init__(self, depth, previous_types=None, FAS=False):
        # Fine-grid Types
        gen_id = lambda base: f"{base}_{depth}"
        self.S_h = self._init_type(gen_id("S"), previous_types, "S_2h")
        self.S_guard_h = self._init_type(gen_id("S_guard"), previous_types, "S_guard_2h", guard=True)
        self.C_h = self._init_type(gen_id("C"), previous_types, "C_2h")
        self.C_guard_h = self._init_type(gen_id("C_guard"), previous_types, "C_guard_2h", guard=True)
        self.x_h = self._init_type(gen_id("x"), previous_types, "x_2h")
        self.A_h = self._init_type(gen_id("A"), previous_types, "A_2h")
        self.B_h = self._init_type(gen_id("A"), previous_types, "B_2h")
        self.R_h = Type(f"R_{depth}")

        # Coarse-Grid Types
        gen_id = lambda base: f"{base}_{depth+1}"
        self.S_2h = Type(gen_id("S"))
        self.S_guard_2h = Type(gen_id("S_guard"), guard=True)
        self.C_2h = Type(gen_id("C"))
        self.C_guard_2h = Type(gen_id("C_guard"), guard=True)
        self.x_2h = Type(gen_id("x"))
        self.A_2h = Type(gen_id("A"))
        self.B_2h = Type(gen_id("B"))
        self.P_2h = Type(gen_id("P"))
        self.CGS_2h = Type(gen_id("CGC"))

        # General Types
        self.Partitioning = self._init_type("Partitioning", previous_types)
        self.RelaxationFactorIndex = self._init_type("RelaxationFactorIndex", previous_types)
        self.BlockShape = self._init_type("BlockShape", previous_types)
        if FAS:
            self.NewtonSteps = self._init_type("NewtonSteps", previous_types)


def add_level(pset, terminals: Terminals, types: Types, depth, coarsest=False, FAS=False):
    if not coarsest:
        coarse_zero_approximation = system.ZeroApproximation(terminals.coarse_grid)
        pset.addTerminal(coarse_zero_approximation, types.x_2h, f'zero_{depth + 1}')
        pset.addTerminal(terminals.coarse_operator, types.A_2h, f'A_{depth + 1}')
    for prolongation in terminals.prolongation_operators:
        pset.addTerminal(prolongation, types.P_2h, prolongation.name)
    for restriction in terminals.restriction_operators:
        pset.addTerminal(restriction, types.R_h, restriction.name)

    scalar_equation = False
    if len(terminals.grid) == 1:
        scalar_equation = True

    # State Transition Functions
    def residual(state):
        approximation, rhs = state
        return base.Cycle(approximation, rhs, base.Residual(terminals.operator, approximation, rhs), predecessor=approximation.predecessor)

    def apply(operator, cycle):
        cycle.correction = base.Multiplication(operator, cycle.correction)
        return cycle

    def update(relaxation_factor_index, partitioning_, cycle):
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

    def update_with_coarse_grid_correction(relaxation_factor_index, prolongation_operator, state, restriction_operator=None):
        cycle = coarse_grid_correction(prolongation_operator, state, restriction_operator)
        return update(relaxation_factor_index, terminals.no_partitioning, cycle)

    def smoothing(relaxation_factor_index, partitioning_, generate_smoother, cycle):
        assert isinstance(cycle.correction, base.Residual), 'Invalid production: expected residual'
        smoothing_operator = generate_smoother(cycle.correction.operator)
        cycle = apply(base.Inverse(smoothing_operator), cycle)
        return update(relaxation_factor_index, partitioning_, cycle)

    def decoupled_jacobi(relaxation_factor_index, partitioning_, cycle):
        return smoothing(relaxation_factor_index, partitioning_, smoother.generate_decoupled_jacobi, cycle)

    def collective_jacobi(relaxation_factor_index, partitioning_, cycle):
        return smoothing(relaxation_factor_index, partitioning_, smoother.generate_collective_jacobi, cycle)

    def collective_block_jacobi(relaxation_factor_index, block_shape, cycle):
        def generate_collective_block_jacobi_fixed(operator):
            return smoother.generate_collective_block_jacobi(operator, block_shape)

        return smoothing(relaxation_factor_index, part.Single, generate_collective_block_jacobi_fixed, cycle)

    def jacobi_picard(relaxation_factor_index, partitioning_, cycle):
        return smoothing(relaxation_factor_index, partitioning_, smoother.generate_jacobi_picard, cycle)

    def jacobi_newton(relaxation_factor_index, partitioning_, n_newton_steps, cycle):
        def generate_jacobi_newton_fixed(operator):
            return smoother.generate_jacobi_newton(operator, n_newton_steps)

        return smoothing(relaxation_factor_index, partitioning_, generate_jacobi_newton_fixed, cycle)
    
    # smoothers in hypre
    def jacobi(relaxation_factor_index, partitioning_, cycle):
        return smoothing(relaxation_factor_index, partitioning_, smoother.generate_jacobi, cycle)
    def GS_forward(relaxation_factor_index, partitioning_, cycle):
        return smoothing(relaxation_factor_index, partitioning_, smoother.generate_GS_forward, cycle)
    def GS_backward(relaxation_factor_index, partitioning_, cycle):
        return smoothing(relaxation_factor_index, partitioning_, smoother.generate_GS_backward, cycle)
    
    # smoothers in hyteg
    def SOR(relaxation_factor_index, partitioning_, cycle):
        return smoothing(relaxation_factor_index, partitioning_, smoother.generate_sor, cycle)
    def SymmtericSOR(relaxation_factor_index, partitioning_, cycle):
        return smoothing(relaxation_factor_index, partitioning_, smoother.generate_symmetricsor, cycle)
    def WeightedJacobi(relaxation_factor_index, partitioning_, cycle):
        return smoothing(relaxation_factor_index, partitioning_, smoother.generate_weightedjacobi, cycle)
    def GaussSeidel(relaxation_factor_index, partitioning_, cycle):
        return smoothing(relaxation_factor_index, partitioning_, smoother.generate_gaussseidel, cycle)
    def Uzawa(relaxation_factor_index, partitioning_, cycle):
        return smoothing(relaxation_factor_index, partitioning_, smoother.generate_uzawa, cycle)
    def SymmetricGaussSeidel(relaxation_factor_index, partitioning_, cycle):
        return smoothing(relaxation_factor_index, partitioning_, smoother.generate_symmetricgaussseidel, cycle)
    def Chebyshev(relaxation_factor_index, partitioning_, cycle):
        return smoothing(relaxation_factor_index, partitioning_, smoother.generate_chebyshev, cycle)
    
    
    def correct_with_coarse_grid_solver(relaxation_factor_index, prolongation_operator, coarse_grid_solver,
                                        restriction_operator, cycle):
        cycle = restrict(restriction_operator, cycle)
        if FAS:
            approximation_c = base.mul(coarse_grid_solver, cycle.correction)
            restricted_solution_FAS = base.mul(restriction_operator, cycle.approximation)
            correction = base.mul(prolongation_operator,
                                  base.sub(approximation_c, restricted_solution_FAS))  # Subtract term for FAS
            cycle.correction = correction
        else:
            cycle = apply(prolongation_operator, apply(coarse_grid_solver, cycle))
        return update(relaxation_factor_index, terminals.no_partitioning, cycle)

    def add_primitive(pset, f, fixed_types, input_types, output_types, name):
        for t1, t2 in zip(input_types, output_types):
            pset.addPrimitive(f, fixed_types + [t1], t2, name)

    # Productions
    add_primitive(pset, residual, [], [types.S_h, types.S_guard_h], [types.C_h, types.C_guard_h], f"residual_{depth}")

    if not scalar_equation:
        add_primitive(pset, decoupled_jacobi, [types.RelaxationFactorIndex, types.Partitioning], [types.C_h, types.C_guard_h], [types.S_h, types.S_guard_h], f"decoupled_jacobi_{depth}")

    if use_hypre:
        add_primitive(pset, jacobi, [types.RelaxationFactorIndex, types.Partitioning], [types.C_h, types.C_guard_h], [types.S_h, types.S_guard_h], f"jacobi_{depth}")
        add_primitive(pset, GS_forward, [types.RelaxationFactorIndex, types.Partitioning], [types.C_h, types.C_guard_h], [types.S_h, types.S_guard_h], f"GS_forward_{depth}")
        add_primitive(pset, GS_backward, [types.RelaxationFactorIndex, types.Partitioning], [types.C_h, types.C_guard_h], [types.S_h, types.S_guard_h], f"GS_backward_{depth}")
    elif use_hyteg:
        # add/remove smoothers for the optimization here.
        add_primitive(pset, SOR, [types.RelaxationFactorIndex, types.Partitioning], [types.C_h, types.C_guard_h], [types.S_h, types.S_guard_h], f"sor_{depth}")
        add_primitive(pset, WeightedJacobi, [types.RelaxationFactorIndex, types.Partitioning], [types.C_h, types.C_guard_h], [types.S_h, types.S_guard_h], f"weightedjacobi_{depth}")
        add_primitive(pset, GaussSeidel, [types.RelaxationFactorIndex, types.Partitioning], [types.C_h, types.C_guard_h], [types.S_h, types.S_guard_h], f"gauseeidel_{depth}")
        add_primitive(pset, SymmetricGaussSeidel, [types.RelaxationFactorIndex, types.Partitioning], [types.C_h, types.C_guard_h], [types.S_h, types.S_guard_h], f"symmetricgaussseidel_{depth}")     
    else:
        # start: Exclude for FAS
        if not FAS:
            add_primitive(pset, collective_jacobi, [types.RelaxationFactorIndex, types.Partitioning], [types.C_h, types.C_guard_h], [types.S_h, types.S_guard_h], f"collective_jacobi_{depth}")
            add_primitive(pset, collective_block_jacobi, [types.RelaxationFactorIndex, types.BlockShape], [types.C_h, types.C_guard_h], [types.S_h, types.S_guard_h], f"collective_block_jacobi_{depth}")
        # end : Exclude for FAS
        if FAS:
            pset.addPrimitive(jacobi_picard, [types.RelaxationFactorIndex, types.Partitioning, types.C_h], types.S_h, f"jacobi_picard_{depth}")
            pset.addPrimitive(jacobi_picard, [types.RelaxationFactorIndex, types.Partitioning, types.C_guard_h], types.S_guard_h, f"jacobi_picard_{depth}")
            pset.addPrimitive(jacobi_newton, [types.RelaxationFactorIndex, types.Partitioning, types.NewtonSteps, types.C_h], types.S_h, f"jacobi_newton_{depth}")
            pset.addPrimitive(jacobi_newton, [types.RelaxationFactorIndex, types.Partitioning, types.NewtonSteps, types.C_guard_h], types.S_guard_h, f"jacobi_newton_{depth}")
            
    if not coarsest:
        if FAS:
            pset.addPrimitive(update_with_coarse_grid_correction,
                              [types.RelaxationFactorIndex, types.P_2h, types.S_2h, types.R_h],
                              types.S_h,
                              f"update_with_coarse_grid_correction_{depth}")
            pset.addPrimitive(update_with_coarse_grid_correction,
                              [types.RelaxationFactorIndex, types.P_2h, types.S_guard_2h, types.R_h],
                              types.S_guard_h,
                              f"update_with_coarse_grid_correction_{depth}")

        else:

            add_primitive(pset, update_with_coarse_grid_correction, [types.RelaxationFactorIndex, types.P_2h], [types.S_2h, types.S_guard_2h], [types.S_h, types.S_guard_h], f"update_with_coarse_grid_correction_{depth}")

        add_primitive(pset, coarsening, [types.A_2h, types.x_2h, types.R_h], [types.C_h, types.C_guard_h], [types.C_2h, types.C_guard_2h], f"coarsening_{depth}")

    else:
        add_primitive(pset, correct_with_coarse_grid_solver, [types.RelaxationFactorIndex, types.P_2h, types.CGS_2h, types.R_h], [types.C_h, types.C_guard_h], [types.S_h, types.S_h], f'correct_with_coarse_grid_solver_{depth}')
        pset.addTerminal(terminals.coarse_grid_solver, types.CGS_2h, f'CGS_{depth + 1}')


def add_block_shapes(pset, fields, approximation, types, dimension, maximum_local_system_size):
    block_shapes = []
    for i in range(len(fields)):
        block_shapes.append([])

        def generate_block_shape(block_shape_, block_shape_max, dimension_):
            if dimension_ == 1:
                for k in range(1, block_shape_max + 1):
                    block_shapes[-1].append(block_shape_ + (k,))
            else:
                for k in range(1, block_shape_max + 1):
                    generate_block_shape(block_shape_ + (k,), block_shape_max, dimension_ - 1)

        generate_block_shape((), maximum_local_system_size, dimension)
    for block_shape_permutation in itertools.product(*block_shapes):
        number_of_terms = 0
        for block_shape in block_shape_permutation:
            number_of_terms += reduce(lambda x, y: x * y, block_shape)
        if len(approximation.grid) < number_of_terms <= maximum_local_system_size:
            pset.addTerminal(block_shape_permutation, types.BlockShape)

def generate_primitive_set(approximation, rhs, dimension, coarsening_factors, max_level, equations, operators, fields,
                           maximum_local_system_size=8, relaxation_factor_samples=37,
                           coarse_grid_solver_expression=None, depth=2, enable_partitioning=True, FAS=False):
    assert depth >= 1, "The maximum number of levels must be greater zero"
    coarsest = False
    if depth == 1:
        coarsest = True
    fine_grid = approximation.grid
    coarse_grid = system.get_coarse_grid(fine_grid, coarsening_factors)
    operator, restriction, prolongation, = \
        generate_operators_from_l2_information(equations, operators, fields, max_level, 0, fine_grid, coarse_grid)
    coarse_operator, coarse_restriction, coarse_prolongation, = \
        generate_operators_from_l2_information(equations, operators, fields, max_level - 1, 1, coarse_grid, system.get_coarse_grid(coarse_grid, coarsening_factors))
    # For now assumes that only one prolongation, restriction and partitioning operator is available
    # TODO: Extend in the future
    partitionings = [part.RedBlack]
    restriction_operators = [restriction]
    prolongation_operators = [prolongation]
    coarse_grid_solver = base.CoarseGridSolver("Coarse-Grid Solver", coarse_operator, coarse_grid_solver_expression)
    relaxation_factor_interval = np.linspace(0.1, 1.9, relaxation_factor_samples)
    terminals = Terminals(approximation, operator, coarse_operator, restriction_operators, prolongation_operators, coarse_grid_solver, relaxation_factor_interval, partitionings)
    types = Types(0, FAS=FAS)
    pset = PrimitiveSetTyped("main", [], types.S_h)
    pset.addTerminal((approximation, rhs), types.S_guard_h, 'u_and_f')
    pset.addTerminal(terminals.no_partitioning, types.Partitioning, terminals.no_partitioning.get_name())
    # Start: Exclude for FAS
    if enable_partitioning:
        for p in terminals.partitionings:
            pset.addTerminal(p, types.Partitioning, p.get_name())
    # End: Exclude for FAS
    for i in range(0, relaxation_factor_samples):
        pset.addTerminal(i, types.RelaxationFactorIndex)

    # Block sizes
    if not FAS:
        add_block_shapes(pset, fields, approximation, types, dimension, maximum_local_system_size)
    # Newton Steps
    if FAS:
        newton_steps = [1, 2, 3, 4]
        for i in newton_steps:
            pset.addTerminal(i, types.NewtonSteps)

    add_level(pset, terminals, types, 0, coarsest=coarsest, FAS=FAS)

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
                generate_system_operator_from_l2_information(equations, operators, fields, max_level - i - 1, i+1, coarse_grid)
        else:
            coarse_operator, coarse_restriction, coarse_prolongation = \
                generate_operators_from_l2_information(equations, operators, fields, max_level - i - 1, i + 1, coarse_grid,
                                                       system.get_coarse_grid(coarse_grid, coarsening_factors))

        coarse_grid_solver = base.CoarseGridSolver("Coarse-Grid Solver", coarse_operator, coarse_grid_solver_expression)
        terminals = Terminals(approximation, operator, coarse_operator, restriction_operators, prolongation_operators, coarse_grid_solver, relaxation_factor_interval, partitionings)
        types_old = types
        types = Types(i, previous_types=types_old, FAS=FAS)
        add_level(pset, terminals, types, i, coarsest=coarsest, FAS=FAS)
        terminal_list.append(terminals)

    return pset, terminal_list
