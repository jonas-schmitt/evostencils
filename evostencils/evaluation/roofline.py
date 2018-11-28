from evostencils.expressions import base, multigrid as mg, partitioning as part
import evostencils.stencils.constant as constant
import evostencils.stencils.periodic as periodic


class RooflineEvaluator:
    """
    Class for estimating the performance of matrix expressions by applying a simple roofline model
    """
    def __init__(self, peak_performance=4*16*2e9, peak_bandwidth=2e10, bytes_per_word=8, coarse_grid_solver_properties=(5, 4),
                 coarse_grid_solver_iterations=100):
        self._peak_performance = peak_performance
        self._peak_bandwidth = peak_bandwidth
        self._bytes_per_word = bytes_per_word
        self._coarse_grid_solver_properties = coarse_grid_solver_properties
        self._coarse_grid_solver_iterations = coarse_grid_solver_iterations

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
    def solver_properties(self):
        return self._coarse_grid_solver_properties

    def compute_performance(self, intensity):
        return min(self.peak_performance, intensity * self.peak_bandwidth)

    def compute_arithmetic_intensity(self, operations, words):
        return operations / (words * self.bytes_per_word)

    def estimate_runtime(self, expression: base.Expression):
        from functools import reduce
        import operator
        list_of_metrics = self.estimate_operations_per_word(expression)
        tmp = ((N * operations, self.compute_arithmetic_intensity(operations, words))
               for operations, words, N in list_of_metrics)
        runtimes = ((total_number_of_operations / self.compute_performance(arithmetic_intensity))
                    for total_number_of_operations, arithmetic_intensity in tmp)
        return reduce(operator.add, runtimes)

    def estimate_operations_per_word(self, expression: base.Expression) -> list:
        result = []
        #if expression.storage is not None and expression.storage.valid:
        #    expression.storage.valid = False

        if isinstance(expression, mg.Cycle):
            if isinstance(expression.correction, base.Multiplication) \
                    and part.can_be_partitioned(expression.correction.operand1):
                stencil_partitions = expression.partitioning.generate(expression.correction.operand1.generate_stencil(), expression.grid)
                for smoother_stencil in stencil_partitions:
                    if isinstance(expression.correction.operand2, mg.Residual):
                        residual = expression.correction.operand2
                        result.extend(self.estimate_operations_per_word(residual.iterate))
                        result.extend(self.estimate_operations_per_word(residual.rhs))
                        #if not residual.rhs.storage.valid:
                        #    result.extend(self.estimate_operations_per_word(residual.rhs))
                        #    residual.rhs.storage.valid = True
                        combined_stencil = periodic.mul(smoother_stencil, residual.operator.generate_stencil())
                        nentries_list1 = periodic.count_number_of_entries(smoother_stencil)
                        nentries_list2 = periodic.count_number_of_entries(combined_stencil)
                        problem_size = expression.shape[0] / max(len(nentries_list1), len(nentries_list2))
                        for nentries1, nentries2 in zip(nentries_list1, nentries_list2):
                            words = self.words_transferred_for_store() + self.words_transferred_for_load() + \
                                    self.words_transferred_for_stencil_application(nentries1) + self.words_transferred_for_stencil_application(nentries2)
                            operations = self.operations_for_addition() + self.operations_for_scaling() + \
                                self.operations_for_subtraction() + self.operations_for_stencil_application(nentries1) + \
                                self.operations_for_stencil_application(nentries2)
                            result.append((words, operations, problem_size))
                    else:
                        result.extend(self.estimate_operations_per_word(expression.correction.operand2))
                        nentries_list = periodic.count_number_of_entries(smoother_stencil)
                        problem_size = expression.shape[0] / len(nentries_list)
                        for nentries in nentries_list:
                            words = self.words_transferred_for_store() + self.words_transferred_for_load() + self.words_transferred_for_stencil_application(nentries)
                            operations = self.operations_for_addition() + self.operations_for_stencil_application(nentries)
                            result.append((words, operations, problem_size))
            else:
                result.extend(self.estimate_operations_per_word(expression.correction))
                words = self.words_transferred_for_store() + 2 * self.words_transferred_for_load()
                operations = self.operations_for_addition()
                result.append((words, operations, expression.shape[0]))

        elif isinstance(expression, mg.Residual):
            result.extend(self.estimate_operations_per_word(expression.iterate))
            result.extend(self.estimate_operations_per_word(expression.rhs))
            #if not expression.rhs.storage.valid:
            #    result.extend(self.estimate_operations_per_word(expression.rhs))
            #    expression.rhs.storage.valid = True
            nentries_list = periodic.count_number_of_entries(expression.operator.generate_stencil())
            for nentries in nentries_list:
                words = self.words_transferred_for_store() + self.words_transferred_for_load() + self.words_transferred_for_stencil_application(nentries)
                operations = self.operations_for_subtraction() + self.operations_for_stencil_application(nentries)
                result.append((words, operations, expression.shape[0]))
        elif isinstance(expression, base.Multiplication):
            if isinstance(expression.operand1, mg.CoarseGridSolver):
                for _ in range(0, self._coarse_grid_solver_iterations):
                    result.append((*self.solver_properties, expression.shape[0]))
            else:
                stencil = expression.operand1.generate_stencil()
                result.extend(self.estimate_operations_per_word_for_stencil(stencil, expression.shape[0]))
            result.extend(self.estimate_operations_per_word(expression.operand2))
        elif isinstance(expression, base.Grid):
            pass
        else:
            print(type(expression))
            raise NotImplementedError("Case not implemented")
        return result

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
