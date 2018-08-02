from evostencils.expressions import base, multigrid


class RooflineEvaluator:
    def __init__(self, peak_performance, bandwidth, bytes_per_word, solver_properties=(4, 5)):
        self._peak_performance= peak_performance
        self._bandwidth = bandwidth
        self._bytes_per_word = bytes_per_word
        self._solver_properties =  solver_properties

    @property
    def peak_performance(self):
        return self._peak_performance

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def bytes_per_word(self):
        return self._bytes_per_word

    @property
    def bytes_per_word(self):
        return self._bytes_per_word

    @property
    def solver_properties(self):
        return self._solver_properties

    def compute_performance(self, intensity):
        return min(self.peak_performance, intensity * self.bandwidth)

    def compute_arithmetic_intensity(self, operations, words):
        return operations / (words * self.bytes_per_word)

    def estimate_operations_per_word_for_correction(self, correction: multigrid.Correction):
        operator_stencil = correction.operator.generate_stencil()
        operation_count = 2 + len(operator_stencil)
        # read u, read b, write temporary, read u, write temporary
        word_count = 5
        iteration_matrix = correction.iteration_matrix
        evaluated, operations, words = self.estimate_operations_per_word_for_iteration(iteration_matrix)
        if not evaluated:
            operations, words = self.estimate_operations_per_word_for_kernel(iteration_matrix)
        operation_count += operations
        word_count += operations
        return operation_count, word_count

    @staticmethod
    def estimate_operations_per_word_for_kernel(self, expression: base.Expression) -> tuple:
        stencil = expression.expression.generate_stencil()
        stencil_width = len(stencil)
        operations = stencil_width + (stencil_width - 1)
        words = stencil_width + 2
        #if any(all(i == 0 for i in entry[0]) for entry in stencil.entries):
        #    words -= 1
        return operations, words


    def estimate_operations_per_word_for_iteration(self, expression: base.Expression) -> tuple:
        if isinstance(expression, base.BinaryExpression)
            if isinstance(expression.operand1, base.Operator) and not multigrid.is_intergrid_operation(expression.operand1) \
                    and isinstance(expression.operand2, base.Operator) and not multigrid.is_intergrid_operation(expression.operand2):
                return False, None, None
            else:
                result1 = self.estimate_operations_per_word_for_iteration(expression.operand1)
                result2 = self.estimate_operations_per_word_for_iteration(expression.operand2)
                operations = 0
                words = 0
                if result1[0] and result2[0]:
                    return False, None, None

                if result1[0]:
                    operations += result1[1]
                    words += result1[2]
                else:
                    operations1, words1 = self.estimate_operations_per_word_for_kernel(expression.operand1)
                    operations += operations1
                    words += words1

                if result1[1]:
                    operations += result2[1]
                    words += result2[2]
                else:
                    operations2, words2 = self.estimate_operations_per_word_for_kernel(expression.operand2)
                    operations += operations2
                    words += words2
                # (A + B) * u = A * u + B * u => op(A*u) + op(B*u) + 1
                if isinstance(expression, base.Addition) or isinstance(expression, base.Subtraction):
                    operations += 1
                # A * B * u => op(A*u) + op(B*u)
                return True, operations, words
        elif isinstance(expression, base.UnaryExpression):
            operations, words = self.estimate_operations_per_word_for_kernel(expression)
            return True, operations, words
        elif isinstance(expression, base.Scaling):
            result = self.estimate_operations_per_word_for_iteration(expression.operand)
            if result[0]:
                # Here we have to apply the scaling separately to the grid
                # Operations: One multiplication per grid point
                # Words: We have to load (into the cache) and store each grid point once
                return True, result[1] + 1, result[2] + 2
            else:
                return False, None, None
        elif isinstance(expression, multigrid.CoarseGridSolver):
            operations, words = self.solver_properties
            return True, operations, words
        elif isinstance(expression, base.Operator):
            operations, words = self.estimate_operations_per_word_for_kernel(expression)
            return True, operations, words
        else:
            raise NotImplementedError("Not implemented")





















