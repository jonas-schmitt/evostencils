from evostencils.ir.base import Entity


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


def generate_conjugate_gradient(operator, number_of_iterations):
    return KrylovSubspaceMethod('ConjugateGradient', operator, number_of_iterations)


def generate_bicgstab(operator, number_of_iterations):
    return KrylovSubspaceMethod('BiCGStab', operator, number_of_iterations)


def generate_minres(operator, number_of_iterations):
    return KrylovSubspaceMethod('MinRes', operator, number_of_iterations)


def generate_conjugate_residual(operator, number_of_iterations):
    return KrylovSubspaceMethod('ConjugateResidual', operator, number_of_iterations)
