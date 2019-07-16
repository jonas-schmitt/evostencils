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
        total_number_of_operations = problem_size * operations
        arithmetic_intensity = self.compute_arithmetic_intensity(operations, words)
        if arithmetic_intensity > 0.0:
            runtime = total_number_of_operations / self.compute_performance(arithmetic_intensity)
        else:
            runtime = 0.0
        return runtime

    def estimate_runtime(self, expression: base.Expression):
        if expression.runtime is not None:
            return expression.runtime
        if isinstance(expression, multigrid.Cycle):
            if expression.partitioning == partitioning.Single:

            elif expression.partitioning == partitioning.RedBlack:
                if isinstance(expression.correction, base.Multiplication):
                    operand1 = expression.correction.operand1
                    operand2 = expression.correction.operand2
                    if isinstance(operand1, base.Inverse) and isinstance(operand2, multigrid.Residual):

        runtime = 0.0
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
        return number_of_entries * RooflineEvaluator.operations_for_multiplication() + \
               (number_of_entries - 1) * RooflineEvaluator.operations_for_addition()

    @staticmethod
    def operations_for_scaling():
        return 1

    @staticmethod
    def words_transferred_for_stencil_application(number_of_entries):
        return number_of_entries

    @staticmethod
    def words_transferred_for_load():
        return 1

    @staticmethod
    def words_transferred_for_store():
        return 1

    def estimate_runtime_for_operator_application(self, inverse: base.Inverse):
        operator = inverse.operand
        entries = operator.entries
        diagonal_operator = True
        zero_below_diagonal = 0
        for i, row_of_entries in enumerate(entries):
            for j, entry in enumerate(row_of_entries):
                if i != j and not isinstance(entry, base.ZeroOperator) and \
                        not isinstance(entry, multigrid.ZeroProlongation) and \
                        not isinstance(entry, multigrid.ZeroRestriction):
                    diagonal_operator = False
                if i > j and (isinstance(entry, base.ZeroOperator) or isinstance(entry, multigrid.ZeroRestriction)
                              or isinstance(entry, multigrid.ZeroProlongation)):
                    zero_below_diagonal += 1

        if diagonal_operator:
            #TODO Handle simple stencil application
            pass
        else:
            # Handle case where local system must be solved
            grid = operator.grid
            number_of_variables = len(grid)
            offsets = [{} for _ in grid]
            additions = (2*number_of_variables**3 + 3*number_of_variables**2 - 5*number_of_variables)/6.0 - number_of_variables * zero_below_diagonal
            multiplications = additions - number_of_variables * zero_below_diagonal
            divisions = number_of_variables * (number_of_variables + 1) / 2 - zero_below_diagonal
            operations = (additions * RooflineEvaluator.operations_for_addition() +
                         multiplications * RooflineEvaluator.operations_for_multiplication() +
                         divisions * RooflineEvaluator.operations_for_division()) \
                         * min([reduce(lambda x, y: x * y, g.size) for g in grid])
            for row_of_entries in entries:
                for j, entry in enumerate(row_of_entries):
                    stencil = entry.generate_stencil()
                    constant_stencil_list = periodic.get_list_of_entries(stencil)
                    problem_size = reduce(lambda x, y: x * y, grid[j].size)
                    for constant_stencil in constant_stencil_list:
                        number_of_entries = constant_stencil.number_of_entries
                        current_offsets = {}
                        for offset, _ in constant_stencil.entries:
                            tmp = 1.0 / len(constant_stencil_list)
                            if offset in offsets[j]:
                                current_offsets[offset] += tmp
                            else:
                                current_offsets[offset] = tmp
                        for offset, fraction in current_offsets.items():
                            if offset not in offsets[j] or fraction > offsets[j][offset]:
                                offsets[j][offset] = fraction
                        if number_of_entries > 1:
                            operations += RooflineEvaluator.operations_for_stencil_application(number_of_entries - 1) \
                                          * problem_size / len(constant_stencil_list)
            words = 0
            for i, offset_dict in enumerate(offsets):
                problem_size = reduce(lambda x, y: x * y, grid[i].size)
                words += RooflineEvaluator.words_transferred_for_store() * problem_size
                for offset, fraction in offset_dict.items():
                    words += RooflineEvaluator.words_transferred_for_load() * fraction * problem_size
            return self.compute_runtime(operations, words, reduce(lambda x, y: x + y, [reduce(lambda x, y: x * y, g.size) for g in grid]))






    @staticmethod
    def estimate_operations_per_word_for_solving_matrix(number_of_unknowns, problem_size) -> tuple:
        n = number_of_unknowns
        # Gaussian Elimination
        operations = 2.0/3.0 * n * n * n
        words = n * (RooflineEvaluator.words_transferred_for_load() + RooflineEvaluator.words_transferred_for_store())
        return operations, words, float(problem_size) / n

    @staticmethod
    def estimate_operations_per_word_for_stencil(stencil, problem_size) -> list:
        number_of_entries_list = periodic.count_number_of_entries(stencil)
        return [(RooflineEvaluator.operations_for_stencil_application(number_of_entries),
                 RooflineEvaluator.words_transferred_for_stencil_application(number_of_entries) +
                 RooflineEvaluator.words_transferred_for_store(),
                 float(problem_size) / len(number_of_entries_list))
                for number_of_entries in number_of_entries_list]
