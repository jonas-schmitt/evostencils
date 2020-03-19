from evostencils.expressions.base import Entity


class KrylovSubspaceMethod(Entity):
    def __init__(self, name, operator, number_of_iterations):
        self._name = name
        self._shape = operator.shape
        self._operator = operator
        self._number_of_iterations = number_of_iterations
        super().__init__()

    @staticmethod
    def generate_stencil():
        return None

    @property
    def grid(self):
        return self.operator.grid

    @property
    def operator(self):
        return self._operator

    @property
    def number_of_iterations(self):
        return self._number_of_iterations

    def __repr__(self):
        return f'KrylovSubspaceMethod({repr(self.name)}, {repr(self.operator)}, {repr(self.number_of_iterations)})'


class ConjugateGradient(KrylovSubspaceMethod):
    def __init__(self, operator, number_of_iterations):
        super().__init__('ConjugateGradient', operator, number_of_iterations)


class BiCGStab(KrylovSubspaceMethod):
    def __init__(self, operator, number_of_iterations):
        super().__init__('BiCGStab', operator, number_of_iterations)


class MinRes(KrylovSubspaceMethod):
    def __init__(self, operator, number_of_iterations):
        super().__init__('MinRes', operator, number_of_iterations)


class ConjugateResidual(KrylovSubspaceMethod):
    def __init__(self, operator, number_of_iterations):
        super().__init__('ConjugateResidual', operator, number_of_iterations)
