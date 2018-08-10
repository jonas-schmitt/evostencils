from evostencils.expressions import base, multigrid
import evostencils.stencils as stencils


class RooflineEvaluator:
    """
    Class for estimating the performance of matrix expressions by applying a simple roofline model
    """
    def __init__(self, peak_performance=4*16*2e9, peak_bandwidth=2e10, bytes_per_word=8, solver_properties=(4, 5)):
        self._peak_performance = peak_performance
        self._peak_bandwidth = peak_bandwidth
        self._bytes_per_word = bytes_per_word
        self._solver_properties = solver_properties

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
        return self._solver_properties

    def compute_performance(self, intensity):
        return min(self.peak_performance, intensity * self.peak_bandwidth)

    def compute_arithmetic_intensity(self, operations, words):
        return operations / (words * self.bytes_per_word)

    def estimate_runtime(self, expression: base.Expression, ):
        from functools import reduce
        import operator
        list_of_metrics = self.estimate_operations_per_word(expression)
        tmp = ((N * operations, self.compute_arithmetic_intensity(operations, words)) for operations, words, N in list_of_metrics)
        runtimes = ((total_number_of_operations / self.compute_performance(arithmetic_intensity))
                    for total_number_of_operations, arithmetic_intensity in tmp)
        return reduce(operator.add, runtimes)

    def estimate_operations_per_word(self, expression: base.Expression) -> list:
        list_of_metrics = []
        if isinstance(expression, multigrid.Correction):
            list_of_metrics.extend(self._estimate_operations_per_word_for_correction(expression))
            grid = expression.grid
            list_of_metrics.extend(self.estimate_operations_per_word(grid))
            return list_of_metrics
        elif isinstance(expression, base.Grid):
            return list_of_metrics
        else:
            raise NotImplementedError("Not implemented")

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

    def _estimate_operations_per_word_for_correction(self, correction: multigrid.Correction) -> list:
        problem_size = correction.grid.shape[0]
        iteration_matrix = correction.iteration_matrix
        evaluated, list_of_metrics = self._estimate_operations_per_word_for_iteration(iteration_matrix)
        operator_stencil = correction.operator.generate_stencil()
        # u = (1 - omega) * u + omega * correction
        #operations_update = self.operations_for_addition() + 2 * self.operations_for_scaling()
        #words_update = self.words_transferred_for_load() + self.words_transferred_for_store()
        # r = (b - A*x)
        if not evaluated:
            # u = (1 - omega) * u + omega * B * u - B * A * u
            iteration_matrix_stencil = iteration_matrix.generate_stencil()
            combined_stencil = stencils.mul(iteration_matrix_stencil, operator_stencil)
            operations = self.operations_for_stencil_application(iteration_matrix_stencil.number_of_entries) \
                + self.operations_for_stencil_application(combined_stencil.number_of_entries) \
                + self.operations_for_subtraction() + self.operations_for_addition() + 3 * self.operations_for_scaling()
            words = self.words_transferred_for_stencil_application(combined_stencil.number_of_entries) \
                + self.words_transferred_for_stencil_application(iteration_matrix_stencil.number_of_entries) \
                + self.words_transferred_for_load() + self.words_transferred_for_store()
            list_of_metrics.append((operations, words, problem_size))
        else:
            operations_residual = self.operations_for_stencil_application(operator_stencil.number_of_entries) \
                + self.operations_for_subtraction()
            words_residual = self.words_transferred_for_load() \
                + self.words_transferred_for_stencil_application(operator_stencil.number_of_entries) \
                + self.words_transferred_for_store()
            list_of_metrics.append((operations_residual, words_residual, problem_size))
            list_of_metrics[-1] = (list_of_metrics[-1][0] + self.operations_for_addition() + 2 * self.operations_for_scaling(),
                                   list_of_metrics[-1][1] + self.words_transferred_for_load(), problem_size)
        return list_of_metrics

    @staticmethod
    def estimate_operations_per_word_for_stencil(stencil: stencils.Stencil, problem_size) -> tuple:
        number_of_entries = stencil.number_of_entries
        operations = RooflineEvaluator.operations_for_stencil_application(number_of_entries)
        words = RooflineEvaluator.operations_for_stencil_application(number_of_entries) \
            + RooflineEvaluator.words_transferred_for_store()
        # if any(all(i == 0 for i in entry[0]) for entry in stencil.entries):
        #    words -= 1
        return operations, words, problem_size

    def _estimate_operations_per_word_for_iteration(self, expression: base.Expression) -> tuple:
        if isinstance(expression, base.BinaryExpression):
            evaluated1, result1 = self._estimate_operations_per_word_for_iteration(expression.operand1)
            evaluated2, result2 = self._estimate_operations_per_word_for_iteration(expression.operand2)
            metrics = []
            if not evaluated1 and not evaluated2:
                return False, []

            if evaluated1:
                metrics.extend(result1)
            else:
                metrics.append(self.estimate_operations_per_word_for_stencil(expression.operand1.generate_stencil(),
                                                                             expression.operand1.shape[0]))

            if evaluated2:
                metrics.extend(result2)
            else:
                metrics.append(self.estimate_operations_per_word_for_stencil(expression.operand2.generate_stencil(),
                               expression.operand2.shape[0]))
            # (A + B) * u = A * u + B * u => op(A*u) + op(B*u) + 1
            if isinstance(expression, base.Addition):
                metrics = [(operations + self.operations_for_addition(), words, problem_size) for operations, words, problem_size in metrics]
            elif isinstance(expression, base.Subtraction):
                metrics = [(operations + self.operations_for_subtraction(), words, problem_size) for operations, words, problem_size in metrics]
            # A * B * u => op(A*u) + op(B*u)
            return True, metrics
        elif isinstance(expression, base.UnaryExpression):
            return False, []
        elif isinstance(expression, base.Scaling):
            result = self._estimate_operations_per_word_for_iteration(expression.operand)
            if result[0]:
                # Here we have to apply the scaling separately to the grid
                # Operations: One multiplication per grid point
                # Words: We have to load (into the cache) and store each grid point once
                metrics = [(operations + self.operations_for_scaling(), words + self.words_transferred_for_load() +
                            self.words_transferred_for_store(), problem_size)
                           for operations, words, problem_size in result]
                return True, metrics
            else:
                return False, []
        elif isinstance(expression, multigrid.CoarseGridSolver):
            return True, [(*self.solver_properties, expression.shape[0])]
        elif isinstance(expression, base.Operator):
            metrics = self.estimate_operations_per_word_for_stencil(expression.generate_stencil(), expression.shape[0])
            return True, [metrics]
        else:
            raise NotImplementedError("Not implemented")





















