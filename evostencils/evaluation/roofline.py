from evostencils.expressions import base, multigrid as mg, partitioning as part
import evostencils.stencils.constant as constant
import evostencils.stencils.periodic as periodic


class RooflineEvaluator:
    """
    Class for estimating the performance of matrix expressions by applying a simple roofline model
    """
    def __init__(self, peak_performance=4*16*2e9, peak_bandwidth=2e10, bytes_per_word=8, runtime_coarse_grid_solver=0):
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
        runtime = 0.0
        if isinstance(expression, mg.Cycle):
            if isinstance(expression.correction, base.Multiplication) \
                    and part.can_be_partitioned(expression.correction.operand1):
                smoother_stencil = expression.correction.operand1.generate_stencil()
                stencil_partitions = expression.partitioning.generate(smoother_stencil, expression.grid)
                for partition in stencil_partitions:
                    if isinstance(expression.correction.operand2, mg.Residual):
                        residual = expression.correction.operand2
                        runtime += self.estimate_runtime(residual.iterate)
                        if hasattr(residual.rhs, 'evaluated'):
                            if not residual.rhs.evaluated:
                                runtime += self.estimate_runtime(residual.rhs)
                                residual.rhs.evaluated = True
                        combined_stencil = periodic.mul(partition, periodic.mul(smoother_stencil, residual.operator.generate_stencil()))
                        nentries_list1 = periodic.count_number_of_entries(periodic.mul(partition, smoother_stencil))
                        nentries_list2 = periodic.count_number_of_entries(combined_stencil)
                        problem_size = expression.shape[0] / max(len(nentries_list1), len(nentries_list2))
                        for nentries1, nentries2 in zip(nentries_list1, nentries_list2):
                            if nentries1 > 0 or nentries2 > 0:
                                words = self.words_transferred_for_store() + self.words_transferred_for_load() + \
                                        self.words_transferred_for_stencil_application(nentries1) + self.words_transferred_for_stencil_application(nentries2)
                                operations = self.operations_for_addition() + self.operations_for_scaling() + \
                                    self.operations_for_subtraction() + self.operations_for_stencil_application(nentries1) + \
                                    self.operations_for_stencil_application(nentries2)
                                runtime += self.compute_runtime(operations, words, problem_size)
                    else:
                        runtime += self.estimate_runtime(expression.correction.operand2)
                        nentries_list = periodic.count_number_of_entries(smoother_stencil)
                        problem_size = expression.shape[0] / len(nentries_list)
                        for nentries in nentries_list:
                            words = self.words_transferred_for_store() + self.words_transferred_for_load() + self.words_transferred_for_stencil_application(nentries)
                            operations = self.operations_for_addition() + self.operations_for_stencil_application(nentries)
                            runtime += self.compute_runtime(operations, words, problem_size)
            else:
                runtime += self.estimate_runtime(expression.correction)
                words = self.words_transferred_for_store() + 2 * self.words_transferred_for_load()
                operations = self.operations_for_addition()
                runtime += self.compute_runtime(operations, words, expression.shape[0])

        elif isinstance(expression, mg.Residual):
            runtime += self.estimate_runtime(expression.iterate)
            runtime += self.estimate_runtime(expression.rhs)
            if hasattr(expression.rhs, 'evaluated'):
                if not expression.rhs.evaluated:
                    runtime += self.estimate_runtime(expression.rhs)
                    expression.rhs.evaluated = True
            nentries_list = periodic.count_number_of_entries(expression.operator.generate_stencil())
            for nentries in nentries_list:
                words = self.words_transferred_for_store() + self.words_transferred_for_load() + self.words_transferred_for_stencil_application(nentries)
                operations = self.operations_for_subtraction() + self.operations_for_stencil_application(nentries)
                runtime += self.compute_runtime(operations, words, expression.shape[0])
        elif isinstance(expression, base.Multiplication):
            if isinstance(expression.operand1, mg.CoarseGridSolver):
                if expression.operand1.expression is not None:
                    if expression.operand1.expression.runtime is None:
                        raise RuntimeError("Not evaluated")
                    runtime += expression.operand1.expression.runtime
            else:
                stencil = expression.operand1.generate_stencil()
                list_of_metrics = self.estimate_operations_per_word_for_stencil(stencil, expression.shape[0])
                for operations, words, problem_size in list_of_metrics:
                    runtime += self.compute_runtime(operations, words, problem_size)
            runtime += self.estimate_runtime(expression.operand2)
        elif isinstance(expression, base.Grid):
            pass
        else:
            print(type(expression))
            raise NotImplementedError("Case not implemented")
        expression.runtime = runtime
        return runtime

    @staticmethod
    def operations_for_addition():
        return 1

    @staticmethod
    def operations_for_subtraction():
        return 1

    @staticmethod
    def operations_for_stencil_application(number_of_entries):
        return number_of_entries + (number_of_entries - 1)

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
