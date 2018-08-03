from evostencils.expressions import base, multigrid
import evostencils.stencils as stencils


class RooflineEvaluator:
    def __init__(self, bytes_per_word, solver_properties=(4, 5)):
        self._bytes_per_word = bytes_per_word
        self._solver_properties =  solver_properties

    @property
    def bytes_per_word(self):
        return self._bytes_per_word

    @property
    def solver_properties(self):
        return self._solver_properties

    @staticmethod
    def compute_performance(self, peak_performance, intensity, bandwidth):
        return min(peak_performance, intensity * bandwidth)

    def compute_arithmetic_intensity(self, operations, words):
        return operations / (words * self.bytes_per_word)

    def compute_average_arithmetic_intensity(self, list_of_metrics: list):
        from functools import reduce
        mean = reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]), list_of_metrics)
        return self.compute_arithmetic_intensity(*mean)

    def compute_maximum_arithmetic_intensity(self, list_of_metrics: list):
        return max([self.compute_arithmetic_intensity(operations, words) for operations, words in list_of_metrics])

    def compute_minimum_arithmetic_intensity(self, list_of_metrics: list):
        return min([self.compute_arithmetic_intensity(operations, words) for operations, words in list_of_metrics])

    def estimate_performance_metrics(self, expression: base.Expression) -> list:
        list_of_metrics = []
        if isinstance(expression, multigrid.Correction):
            list_of_metrics.extend(self._estimate_correction_performance_metrics(expression))
            grid = expression.grid
            list_of_metrics.extend(self.estimate_performance_metrics(grid))
            return list_of_metrics
        elif isinstance(expression, base.Grid):
            return list_of_metrics
        else:
            raise NotImplementedError("Not implemented")

    def _estimate_correction_performance_metrics(self, correction: multigrid.Correction) -> list:
        iteration_matrix = correction.iteration_matrix
        evaluated, list_of_metrics = self._estimate_iteration_performance_metrics(iteration_matrix)
        operator_stencil = correction.operator.generate_stencil()
        number_of_entries = operator_stencil.number_of_entries
        operations_update = 1
        words_update = 1
        operations_residual = number_of_entries + (number_of_entries - 1) + 1
        words_residual = number_of_entries + 3
        if not evaluated:
            stencil = iteration_matrix.generate_stencil()
            list_of_metrics.append(self.estimate_stencil_application_performance_metrics(stencil))
            list_of_metrics[-1] = (operations_residual + list_of_metrics[-1][0] + operations_update,
                                   operations_residual + list_of_metrics[-1][1] + words_update)
        else:
            list_of_metrics.append((operations_residual + operations_update, words_residual + words_update + 2))
        # r = b - Au
        # tmp = iteration_matrix * r
        # u = u + tmp
        return list_of_metrics


    @staticmethod
    def estimate_stencil_application_performance_metrics(stencil: stencils.Stencil) -> tuple:
        number_of_entries = stencil.number_of_entries
        operations = number_of_entries + (number_of_entries - 1)
        words = number_of_entries + 2
        # if any(all(i == 0 for i in entry[0]) for entry in stencil.entries):
        #    words -= 1
        return operations, words

    def _estimate_iteration_performance_metrics(self, expression: base.Expression) -> tuple:
        if isinstance(expression, base.BinaryExpression):
            evaluated1, result1 = self._estimate_iteration_performance_metrics(expression.operand1)
            evaluated2, result2 = self._estimate_iteration_performance_metrics(expression.operand2)
            metrics = []
            if not evaluated1 and not evaluated2:
                return False, []

            if evaluated1:
                metrics.extend(result1)
            else:
                metrics.append(self.estimate_stencil_application_performance_metrics(expression.operand1.generate_stencil()))

            if evaluated2:
                metrics.extend(result2)
            else:
                metrics.append(self.estimate_stencil_application_performance_metrics(expression.operand2.generate_stencil()))
            # (A + B) * u = A * u + B * u => op(A*u) + op(B*u) + 1
            if isinstance(expression, base.Addition) or isinstance(expression, base.Subtraction):
                metrics = [(operations + 1, words) for operations, words in metrics]
            # A * B * u => op(A*u) + op(B*u)
            return True, metrics
        elif isinstance(expression, base.UnaryExpression):
            return False, []
        elif isinstance(expression, base.Scaling):
            result = self._estimate_iteration_performance_metrics(expression.operand)
            if result[0]:
                # Here we have to apply the scaling separately to the grid
                # Operations: One multiplication per grid point
                # Words: We have to load (into the cache) and store each grid point once
                metrics = [(operations + 1, words + 2) for operations, words in result]
                return True, metrics
            else:
                return False, []
        elif isinstance(expression, multigrid.CoarseGridSolver):
            return True, [self.solver_properties]
        elif isinstance(expression, base.Operator):
            metrics = self.estimate_stencil_application_performance_metrics(expression.generate_stencil())
            return True, [metrics]
        else:
            raise NotImplementedError("Not implemented")





















