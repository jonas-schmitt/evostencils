from evostencils.expressions import base, multigrid, partitioning, system
import evostencils.stencils.periodic as periodic
from functools import reduce


class RooflineEvaluator:
    """
    Class for estimating the performance of matrix expressions by applying a simple roofline model
    """
    def __init__(self, peak_performance, peak_bandwidth, bytes_per_word, runtime_coarse_grid_solver=0):
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

    def set_runtime_of_coarse_grid_solver(self, runtime_coarse_grid_solver):
        self._runtime_coarse_grid_solver = runtime_coarse_grid_solver

    def compute_performance(self, intensity):
        return min(self.peak_performance, intensity * self.peak_bandwidth)

    def compute_arithmetic_intensity(self, operations, words):
        return operations / (words * self.bytes_per_word)

    def compute_runtime(self, operations, words, problem_size):
        arithmetic_intensity = self.compute_arithmetic_intensity(operations, words)
        if arithmetic_intensity > 0.0:
            runtime = problem_size / self.compute_performance(arithmetic_intensity)
        else:
            runtime = 0.0
        return runtime

    def estimate_runtime(self, expression: base.Expression):
        if expression.runtime is not None:
            return expression.runtime
        if isinstance(expression, multigrid.Cycle):
            # Partitions are currently ignored since they do not affect the arithmetic intensity
            # Although beware that in cases where the total data size is too the memory bandwidth can not be saturated
            # and the maximum bandwidth will not be achieved
            grid = expression.grid
            correction = expression.correction
            if not isinstance(correction, base.Multiplication):
                raise RuntimeError("Expected multiplication")
            if isinstance(correction.operand1, system.InterGridOperator):
                operations_per_cell, words_per_cell = RooflineEvaluator.estimate_words_per_operation_for_intergrid_transfer(correction.operand1)
            else:
                operations_per_cell, words_per_cell = RooflineEvaluator.estimate_words_per_operation_for_solving_local_system(correction.operand1)
            operations_per_cell += len(grid) * (RooflineEvaluator.operations_for_addition() + RooflineEvaluator.operations_for_scaling())
            words_per_cell += len(grid) * (RooflineEvaluator.words_transferred_for_load() + RooflineEvaluator.words_transferred_for_store())
            problem_size = min([reduce(lambda x, y: x * y, g.size) for g in expression.grid])
            runtime = self.compute_runtime(operations_per_cell, words_per_cell, problem_size)
            return runtime + self.estimate_runtime(correction.operand2)
        elif isinstance(expression, multigrid.Residual):
            operations_per_cell, words_per_cell = RooflineEvaluator.estimate_words_per_operation_for_residual(expression)
            # Store result
            words_per_cell += len(expression.grid) * RooflineEvaluator.words_transferred_for_store()
            problem_size = min([reduce(lambda x, y: x * y, g.size) for g in expression.grid])
            runtime = self.compute_runtime(operations_per_cell, words_per_cell, problem_size)
            return runtime + self.estimate_runtime(expression.rhs) + self.estimate_runtime(expression.approximation)
        elif isinstance(expression, base.Multiplication):
            if isinstance(expression.operand1, system.InterGridOperator):
                operations_per_cell, words_per_cell = RooflineEvaluator.estimate_words_per_operation_for_intergrid_transfer(expression.operand1)
            elif isinstance(expression.operand1, multigrid.CoarseGridSolver):
                cgs = expression.operand1
                if cgs.expression is not None:
                    if cgs.expression.runtime is None:
                        raise RuntimeError("Not evaluated")
                    runtime = cgs.expression.runtime
                else:
                    runtime = self.runtime_coarse_grid_solver
                return runtime + self.estimate_runtime(expression.operand2)
            else:
                operations_per_cell, words_per_cell = RooflineEvaluator.estimate_words_per_operation_for_solving_local_system(expression.operand1)
            problem_size = min([reduce(lambda x, y: x * y, g.size) for g in expression.grid])
            runtime = self.compute_runtime(operations_per_cell, words_per_cell, problem_size)
            return runtime + self.estimate_runtime(expression.operand2)
        else:
            RuntimeError("Not implemented")

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
        return number_of_entries * RooflineEvaluator.operations_for_multiplication() + \
               (number_of_entries - 1) * RooflineEvaluator.operations_for_addition()

    @staticmethod
    def operations_for_scaling():
        return 1

    @staticmethod
    def words_transferred_for_stencil_application(number_of_entries):
        return number_of_entries * RooflineEvaluator.words_transferred_for_load()

    @staticmethod
    def words_transferred_for_load():
        return 1

    @staticmethod
    def words_transferred_for_store():
        return 1

    @staticmethod
    def estimate_words_per_operation_for_residual(residual: multigrid.Residual):
        grid = residual.grid
        offset_sets = [set() for _ in grid]
        operator = residual.operator
        operations_per_cell = 0
        # Load right-hand side
        words_per_cell = len(grid) * RooflineEvaluator.words_transferred_for_load()
        for row_of_entries in operator.entries:
            for i, entry in enumerate(row_of_entries):
                stencil = entry.generate_stencil()
                list_of_entries = periodic.get_list_of_entries(stencil)
                first_constant_stencil = list_of_entries[0]
                assert all(all(coeff1[0] == coeff2[0] for coeff1, coeff2 in zip(first_constant_stencil, constant_stencil)) for constant_stencil in list_of_entries), \
                    'The offsets must be the same for all operator stencils'
                constant_stencil = first_constant_stencil
                for offset, _ in constant_stencil.entries:
                    offset_sets[i].add(offset)
                number_of_stencil_coefficients = constant_stencil.number_of_entries
                operations_per_cell += RooflineEvaluator.operations_for_stencil_application(number_of_stencil_coefficients) + RooflineEvaluator.operations_for_subtraction()
        for s in offset_sets:
            words_per_cell += RooflineEvaluator.words_transferred_for_stencil_application(len(s))
        return operations_per_cell, words_per_cell

    @staticmethod
    def estimate_words_per_operation_for_solving_local_system(inverse: base.Inverse):
        # TODO consider variable stencil coefficients that must be loaded additionally
        expression = inverse.operand
        grid = expression.grid
        number_of_variables = len(grid)

        if isinstance(expression, system.Diagonal):
            # Decoupled Relaxation
            operations_per_cell = RooflineEvaluator.operations_for_multiplication() * number_of_variables
            words_per_cell = RooflineEvaluator.words_transferred_for_load() * number_of_variables
        elif isinstance(expression, system.ElementwiseDiagonal):
            # Collective Relaxation
            operator = expression.operand
            entries = operator.entries
            zero_below_diagonal = 0
            for i, row_of_entries in enumerate(entries):
                for j in range(0, i):
                    if isinstance(row_of_entries[j], base.ZeroOperator):
                        zero_below_diagonal += 1

            # Required operations for Gaussian Elimination
            additions = (2*number_of_variables**3 + 3*number_of_variables**2 - 5*number_of_variables)/6.0 - number_of_variables * zero_below_diagonal
            multiplications = additions
            divisions = number_of_variables * (number_of_variables + 1) / 2 - zero_below_diagonal
            operations_per_cell = additions * RooflineEvaluator.operations_for_addition() + \
                multiplications * RooflineEvaluator.operations_for_multiplication() + \
                divisions * RooflineEvaluator.operations_for_division()
            words_per_cell = number_of_variables * RooflineEvaluator.words_transferred_for_load()
        else:
            raise NotImplementedError("Smoother currently not supported.")
        return operations_per_cell, words_per_cell

    @staticmethod
    def estimate_words_per_operation_for_intergrid_transfer(intergrid_operator: system.InterGridOperator):
        operations_per_cell = 0
        words_per_cell = 0
        for row_of_entries in intergrid_operator.entries:
            for entry in row_of_entries:
                if isinstance(entry, multigrid.ZeroProlongation) or isinstance(entry, multigrid.ZeroRestriction):
                    continue
                stencil = entry.generate_stencil()
                list_of_entries = periodic.get_list_of_entries(stencil)
                first_constant_stencil = list_of_entries[0]
                assert all(all(coeff1[0] == coeff2[0] for coeff1, coeff2 in zip(first_constant_stencil, constant_stencil)) for constant_stencil in list_of_entries), \
                    'The offsets must be the same for all operator stencils'
                constant_stencil = first_constant_stencil
                number_of_stencil_coefficients = constant_stencil.number_of_entries
                operations_per_cell += RooflineEvaluator.operations_for_stencil_application(number_of_stencil_coefficients)
                words_per_cell += RooflineEvaluator.words_transferred_for_stencil_application(number_of_stencil_coefficients)
        return operations_per_cell, words_per_cell

