from evostencils.expressions import base, multigrid
import evostencils.stencils as stencils


class RooflineEvaluator:
    def __init__(self, bytes_per_word=8, solver_properties=(4, 5)):
        self._bytes_per_word = bytes_per_word
        self._solver_properties = solver_properties

    @property
    def bytes_per_word(self):
        return self._bytes_per_word

    @property
    def solver_properties(self):
        return self._solver_properties

    @staticmethod
    def compute_performance(peak_performance, intensity, bandwidth):
        return min(peak_performance, intensity * bandwidth)

    def compute_arithmetic_intensity(self, operations, words):
        return operations / (words * self.bytes_per_word)

    def estimate_runtime(self, expression: base.Expression, peak_performance, peak_bandwidth):
        from functools import reduce
        import operator
        list_of_metrics = self.estimate_operations_per_word(expression)
        tmp = ((N * operations, self.compute_arithmetic_intensity(operations, words)) for operations, words, N in list_of_metrics)
        runtimes = ((total_number_of_operations / self.compute_performance(peak_performance, arithmetic_intensity, peak_bandwidth))
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

    def _estimate_operations_per_word_for_correction(self, correction: multigrid.Correction) -> list:
        problem_size = correction.grid.shape[0]
        iteration_matrix = correction.iteration_matrix
        evaluated, list_of_metrics = self._estimate_operations_per_word_for_iteration(iteration_matrix)
        operator_stencil = correction.operator.generate_stencil()
        number_of_entries = operator_stencil.number_of_entries
        operations_update = 1
        words_update = 1
        operations_residual = number_of_entries + (number_of_entries - 1) + 1
        words_residual = number_of_entries + 3
        if not evaluated:
            stencil = iteration_matrix.generate_stencil()
            list_of_metrics.append(self.estimate_operations_per_word_for_stencil(stencil, problem_size))
            list_of_metrics[-1] = (operations_residual + list_of_metrics[-1][0] + operations_update,
                                   operations_residual + list_of_metrics[-1][1] + words_update, list_of_metrics[-1][2])
        else:
            list_of_metrics.append((operations_residual + operations_update, words_residual + words_update + 2,
                                    problem_size))
        # r = b - Au
        # tmp = iteration_matrix * r
        # u = u + tmp
        return list_of_metrics

    @staticmethod
    def estimate_operations_per_word_for_stencil(stencil: stencils.Stencil, problem_size) -> tuple:
        number_of_entries = stencil.number_of_entries
        operations = number_of_entries + (number_of_entries - 1)
        words = number_of_entries + 2
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
                                                                             expression.operand1.shape[1]))

            if evaluated2:
                metrics.extend(result2)
            else:
                metrics.append(self.estimate_operations_per_word_for_stencil(expression.operand2.generate_stencil(),
                               expression.operand2.shape[1]))
            # (A + B) * u = A * u + B * u => op(A*u) + op(B*u) + 1
            if isinstance(expression, base.Addition) or isinstance(expression, base.Subtraction):
                metrics = [(operations + 1, words, problem_size) for operations, words, problem_size in metrics]
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
                metrics = [(operations + 1, words + 2, problem_size) for operations, words, problem_size in result]
                return True, metrics
            else:
                return False, []
        elif isinstance(expression, multigrid.CoarseGridSolver):
            return True, [(*self.solver_properties, expression.shape[1])]
        elif isinstance(expression, base.Operator):
            metrics = self.estimate_operations_per_word_for_stencil(expression.generate_stencil(), expression.shape[1])
            return True, [metrics]
        else:
            raise NotImplementedError("Not implemented")





















