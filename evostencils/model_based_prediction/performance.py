from evostencils.ir import base, partitioning, system
import evostencils.stencils.multiple as periodic
from functools import reduce


class PerformanceEvaluator:
    """
    Class for estimating the performance of matrix ir by applying a simple roofline model
    """
    def __init__(self, peak_performance: float, peak_bandwidth: float, bytes_per_word: int,
                 runtime_coarse_grid_solver=0):
        self._peak_performance = peak_performance
        self._peak_bandwidth = peak_bandwidth
        self._bytes_per_word = bytes_per_word
        self._runtime_coarse_grid_solver = runtime_coarse_grid_solver

    @property
    def peak_performance(self):
        return self._peak_performance

    @property
    def peak_bandwidth(self):
        return self._peak_bandwidth

    @property
    def bytes_per_word(self):
        return self._bytes_per_word

    @property
    def runtime_coarse_grid_solver(self):
        return self._runtime_coarse_grid_solver

    def set_runtime_of_coarse_grid_solver(self, runtime_coarse_grid_solver: float):
        self._runtime_coarse_grid_solver = runtime_coarse_grid_solver

    def compute_performance(self, intensity: float):
        return min(self.peak_performance, intensity * self.peak_bandwidth)

    def compute_arithmetic_intensity(self, operations: float, words: float):
        return operations / (words * self.bytes_per_word)

    def compute_runtime(self, operations: float, words: float, total_number_of_operations: float):
        arithmetic_intensity = self.compute_arithmetic_intensity(operations, words)
        if arithmetic_intensity > 0.0:
            runtime = total_number_of_operations / self.compute_performance(arithmetic_intensity)
        else:
            runtime = 0.0
        return runtime

    def estimate_runtime(self, expression: base.Expression):
        if expression.runtime is not None:
            return expression.runtime
        if isinstance(expression, base.Cycle):
            # Partitions are currently ignored since they do not affect the arithmetic intensity
            # Although beware that in cases where the total data size is too small
            # the memory bandwidth can not be saturated
            # and the maximum bandwidth will not be achieved
            grid = expression.grid
            correction = expression.correction
            if isinstance(correction, base.Residual):
                operations_per_cell, words_per_cell = 0, 0
                runtime = self.estimate_runtime(correction)
            elif isinstance(correction, base.Multiplication):
                if isinstance(correction.operand1, system.InterGridOperator):
                    # Estimate runtime for right-hand side of expression
                    runtime = self.estimate_runtime(correction.operand2)
                    operations_per_cell, words_per_cell = \
                        PerformanceEvaluator.estimate_words_per_operation_for_intergrid_transfer(correction.operand1)
                else:
                    residual = correction.operand2
                    # Estimate runtime for approximation and rhs in residual
                    # The residual is considered within solving the local system
                    if not isinstance(residual.rhs, system.RightHandSide):
                        runtime_rhs = self.estimate_runtime(residual.rhs)
                    else:
                        runtime_rhs = 0
                    if not isinstance(residual.approximation, system.Approximation):
                        runtime_approximation = self.estimate_runtime(residual.approximation)
                    else:
                        runtime_approximation = 0
                    runtime = runtime_rhs + runtime_approximation
                    operations_per_cell, words_per_cell = \
                        PerformanceEvaluator.estimate_words_per_operation_for_solving_local_system(correction.operand1, residual)
            else:
                raise RuntimeError("Expected multiplication")
            operations_per_cell += len(grid) * (PerformanceEvaluator.operations_for_addition()
                                                + PerformanceEvaluator.operations_for_scaling())
            words_per_cell += len(grid) * (PerformanceEvaluator.words_transferred_for_load()
                                           + PerformanceEvaluator.words_transferred_for_store())
            problem_size = min([reduce(lambda x, y: x * y, g.size) for g in expression.grid])
            # Compute the total runtime for solving the local system and computing a new approximation
            tmp = self.compute_runtime(operations_per_cell, words_per_cell, operations_per_cell * problem_size)
            if expression.partitioning == partitioning.RedBlack:
                tmp *= 1.4303682894270744 # Experimentally obtained factor
            runtime += tmp
        elif isinstance(expression, base.Residual):
            # Estimate runtime for approximation and rhs in residual
            if not isinstance(expression.rhs, system.RightHandSide):
                runtime_rhs = self.estimate_runtime(expression.rhs)
            else:
                runtime_rhs = 0
            if not isinstance(expression.approximation, system.Approximation):
                runtime_approximation = self.estimate_runtime(expression.approximation)
            else:
                runtime_approximation = 0
            runtime = runtime_rhs + runtime_approximation
            operations_per_cell, words_per_cell = PerformanceEvaluator.estimate_words_per_operation_for_residual(expression)
            # Store result
            words_per_cell += len(expression.grid) * PerformanceEvaluator.words_transferred_for_store()
            problem_size = min([reduce(lambda x, y: x * y, g.size) for g in expression.grid])
            runtime += self.compute_runtime(operations_per_cell, words_per_cell, operations_per_cell * problem_size)
        elif isinstance(expression, base.Multiplication):
            if isinstance(expression.operand1, system.InterGridOperator):
                runtime = self.estimate_runtime(expression.operand2)
                operations_per_cell, words_per_cell = \
                    PerformanceEvaluator.estimate_words_per_operation_for_intergrid_transfer(expression.operand1)
                problem_size = min([reduce(lambda x, y: x * y, g.size) for g in expression.grid])
                runtime += self.compute_runtime(operations_per_cell, words_per_cell, operations_per_cell * problem_size)
            elif isinstance(expression.operand1, base.CoarseGridSolver):
                runtime = self.estimate_runtime(expression.operand2)
                coarse_grid_solver = expression.operand1
                if coarse_grid_solver.expression is not None:
                    if coarse_grid_solver.expression.runtime is None:
                        runtime += self.estimate_runtime(coarse_grid_solver.expression)
                    else:
                        runtime += coarse_grid_solver.expression.runtime
                else:
                    runtime += self.runtime_coarse_grid_solver
            else:
                residual = expression.operand2
                if not isinstance(residual.rhs, system.RightHandSide):
                    runtime_rhs = self.estimate_runtime(residual.rhs)
                else:
                    runtime_rhs = 0
                if not isinstance(residual.approximation, system.Approximation):
                    runtime_approximation = self.estimate_runtime(residual.approximation)
                else:
                    runtime_approximation = 0
                runtime = runtime_rhs + runtime_approximation
                operations_per_cell, words_per_cell = \
                    PerformanceEvaluator.estimate_words_per_operation_for_solving_local_system(expression.operand1,
                                                                                               residual)
                problem_size = min([reduce(lambda x, y: x * y, g.size) for g in expression.grid])
                runtime += self.compute_runtime(operations_per_cell, words_per_cell, operations_per_cell * problem_size)
        else:
            raise RuntimeError("Not implemented")
        expression.runtime = runtime
        return runtime

    @staticmethod
    def operations_for_addition():
        return 1

    @staticmethod
    def operations_for_multiplication():
        return 1

    @staticmethod
    def operations_for_division():
        return 1

    @staticmethod
    def operations_for_subtraction():
        return 1

    @staticmethod
    def operations_for_stencil_application(number_of_entries):
        return number_of_entries * PerformanceEvaluator.operations_for_multiplication() + \
               (number_of_entries - 1) * PerformanceEvaluator.operations_for_addition()

    @staticmethod
    def operations_for_scaling():
        return 1

    @staticmethod
    def words_transferred_for_stencil_application(number_of_entries):
        return number_of_entries * PerformanceEvaluator.words_transferred_for_load()

    @staticmethod
    def words_transferred_for_load():
        return 1

    @staticmethod
    def words_transferred_for_store():
        return 1

    @staticmethod
    def estimate_words_per_operation_for_residual(residual: base.Residual):
        grid = residual.grid
        offset_sets = [set() for _ in grid]
        operator = residual.operator
        operations_per_cell = 0
        # Load right-hand side
        words_per_cell = len(grid) * PerformanceEvaluator.words_transferred_for_load()
        for row_of_entries in operator.entries:
            for i, entry in enumerate(row_of_entries):
                stencil = entry.generate_stencil()
                list_of_entries = periodic.get_list_of_entries(stencil)
                first_constant_stencil = list_of_entries[0]
                assert all(all(coeff1[0] == coeff2[0] for coeff1, coeff2 in zip(first_constant_stencil.entries, constant_stencil.entries)) for constant_stencil in list_of_entries), \
                    'The offsets must be the same for all operator stencils'
                constant_stencil = first_constant_stencil
                for offset, _ in constant_stencil.entries:
                    offset_sets[i].add(offset)
                number_of_stencil_coefficients = constant_stencil.number_of_entries
                operations_per_cell += PerformanceEvaluator.operations_for_stencil_application(number_of_stencil_coefficients) \
                    + PerformanceEvaluator.operations_for_subtraction()
        for s in offset_sets:
            words_per_cell += PerformanceEvaluator.words_transferred_for_stencil_application(len(s))
        words_per_cell += len(grid) * PerformanceEvaluator.words_transferred_for_store()
        return operations_per_cell, words_per_cell

    @staticmethod
    def estimate_words_per_operation_for_solving_local_system(inverse: base.Inverse, residual: base.Residual):
        # TODO consider variable stencil coefficients that must be loaded additionally
        expression = inverse.operand
        grid = expression.grid
        number_of_variables = len(grid)

        operations_per_cell_residual, words_per_cell_residual = \
            PerformanceEvaluator.estimate_words_per_operation_for_residual(residual)
        if isinstance(expression, system.Diagonal):
            # Decoupled Relaxation
            operations_per_cell = PerformanceEvaluator.operations_for_multiplication() * number_of_variables \
                                  + operations_per_cell_residual
            words_per_cell = PerformanceEvaluator.words_transferred_for_load() * number_of_variables \
                + words_per_cell_residual
        elif isinstance(expression, system.ElementwiseDiagonal) or isinstance(expression, system.Operator):
            # Collective Relaxation
            # Required operations for Gaussian Elimination
            if isinstance(expression, system.Operator):
                entries = expression.entries
                for i in range(len(grid)):
                    entry = entries[i][i]
                    stencil = entry.generate_stencil()
                    number_of_entries = periodic.count_number_of_entries(stencil)
                    number_of_additional_variables = len(number_of_entries) - 1
                    number_of_variables += number_of_additional_variables

            n = number_of_variables
            multiplications = int(round(n**3 / 3 + n**2 - n / 3))
            additions = int(round(n**3 / 3 + n**2 / 2 - 5*n / 6))
            assert n % len(grid) == 0, "Wrong number of variables in solve locally"
            operations_per_cell = additions * PerformanceEvaluator.operations_for_addition() + \
                multiplications * PerformanceEvaluator.operations_for_multiplication() \
                + n // len(grid) * operations_per_cell_residual
            words_per_cell = number_of_variables * PerformanceEvaluator.words_transferred_for_load() \
                + n // len(grid) * words_per_cell_residual
        else:
            raise NotImplementedError("Smoother currently not supported.")
        return operations_per_cell, words_per_cell

    @staticmethod
    def estimate_words_per_operation_for_intergrid_transfer(intergrid_operator: system.InterGridOperator):
        operations_per_cell = 0
        words_per_cell = 0
        for row_of_entries in intergrid_operator.entries:
            for entry in row_of_entries:
                if isinstance(entry, base.ZeroProlongation) or isinstance(entry, base.ZeroRestriction):
                    continue
                stencil = entry.generate_stencil()
                list_of_entries = periodic.get_list_of_entries(stencil)
                first_constant_stencil = list_of_entries[0]
                assert all(all(coeff1[0] == coeff2[0] for coeff1, coeff2 in zip(first_constant_stencil.entries, constant_stencil.entries)) for constant_stencil in list_of_entries), \
                    'The offsets must be the same for all operator stencils'
                constant_stencil = first_constant_stencil
                number_of_stencil_coefficients = constant_stencil.number_of_entries
                operations_per_cell += PerformanceEvaluator.operations_for_stencil_application(number_of_stencil_coefficients)
                words_per_cell += PerformanceEvaluator.words_transferred_for_stencil_application(number_of_stencil_coefficients)
        return operations_per_cell, words_per_cell

