class OperatorType:
    def __init__(self, shape, diagonal=True, inter_grid_operator=False):
        self.shape = shape
        self.diagonal = diagonal
        self.inter_grid_operator = inter_grid_operator

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.shape == other.shape and self.diagonal == other.diagonal and \
                   self.inter_grid_operator == other.inter_grid_operator
        else:
            return False

    def issubtype(self, other):
        if isinstance(other, type(self)):
            is_subtype = True
            if self.shape != other.shape:
                return False
            if not self.diagonal:
                is_subtype = is_subtype and not other.diagonal
            if not self.inter_grid_operator:
                is_subtype = is_subtype and not other.inter_grid_operator
            return is_subtype
        else:
            return False

    def __hash__(self):
        return hash((type(self), *self.shape, self.diagonal))


def generate_system_operator_type(shape):
    return OperatorType(shape, diagonal=False, inter_grid_operator=False)


def generate_inter_grid_operator_type(shape):
    return OperatorType(shape, diagonal=False, inter_grid_operator=True)


class SolverType:
    def __init__(self, shape):
        self.shape = shape

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.shape == other.shape
        else:
            return False

    def issubtype(self, other):
        if isinstance(other, type(self)):
            return self.shape == other.shape
        else:
            return False

    def __hash__(self):
        return hash((type(self), *self.shape))


def generate_solver_type(shape):
    return SolverType(shape)
