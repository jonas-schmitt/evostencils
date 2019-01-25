class OperatorType:
    def __init__(self, shape, diagonal=True, block_diagonal=True, lower_triangle=True, upper_triangle=True):
        self.shape = shape
        self.diagonal = diagonal
        self.block_diagonal = block_diagonal
        self.lower_triangle = lower_triangle
        self.upper_triangle = upper_triangle

    def __eq__(self, other):
        if hasattr(other, 'shape') and hasattr(other, 'diagonal') and hasattr(other, 'block_diagonal') \
                and hasattr(other, 'lower_triangle') \
                and hasattr(other, 'upper_triangle'):
            return self.shape == other.shape and self.diagonal == other.diagonal and \
                   self.block_diagonal == other.block_diagonal and self.lower_triangle == other.lower_triangle and \
                   self.upper_triangle == other.upper_triangle
        else:
            return False

    def issubtype(self, other):
        if hasattr(other, 'shape') and hasattr(other, 'diagonal') and hasattr(other, 'block_diagonal') \
                and hasattr(other, 'lower_triangle') \
                and hasattr(other, 'upper_triangle'):
            is_subtype = True
            if self.shape != other.shape:
                return False
            if not self.diagonal:
                is_subtype = is_subtype and not other.diagonal
            if not self.lower_triangle:
                is_subtype = is_subtype and not other.lower_triangle
            if not self.upper_triangle:
                is_subtype = is_subtype and not other.upper_triangle
            if not self.block_diagonal:
                is_subtype = is_subtype and not other.block_diagonal
            return is_subtype
        else:
            return False

    def __hash__(self):
        return hash((*self.shape, self.diagonal, self.block_diagonal, self.lower_triangle, self.upper_triangle))


def generate_operator_type(shape):
    return OperatorType(shape, diagonal=True, block_diagonal=True, lower_triangle=True, upper_triangle=True)


def generate_diagonal_operator_type(shape):
    return OperatorType(shape, diagonal=True, block_diagonal=False, lower_triangle=False, upper_triangle=False)


def generate_block_diagonal_operator_type(shape):
    return OperatorType(shape, diagonal=True, block_diagonal=True, lower_triangle=False, upper_triangle=False)


def generate_strictly_lower_triangular_operator_type(shape):
    return OperatorType(shape, diagonal=False, block_diagonal=False, lower_triangle=True, upper_triangle=False)


def generate_strictly_upper_triangular_operator_type(shape):
    return OperatorType(shape, diagonal=False, block_diagonal=False, lower_triangle=False, upper_triangle=True)


def generate_lower_triangular_operator_type(shape):
    return OperatorType(shape, diagonal=True, block_diagonal=False, lower_triangle=True, upper_triangle=False)


def generate_upper_triangular_operator_type(shape):
    return OperatorType(shape, diagonal=True, block_diagonal=False, lower_triangle=False, upper_triangle=True)


def generate_zero_operator_type(shape):
    return OperatorType(shape, diagonal=False, block_diagonal=False, lower_triangle=False, upper_triangle=False)


class SolverType:
    def __init__(self, shape):
        self.shape = shape
        self.is_solver = True

    def __eq__(self, other):
        if hasattr(other, 'shape') and hasattr(other, 'is_solver'):
            return self.shape == other.shape and self.is_solver == other.is_solver
        else:
            return False

    def issubtype(self, other):
        if hasattr(other, 'shape') and hasattr(other, 'is_solver'):
            return self.shape == other.shape and self.is_solver == other.is_solver
        else:
            return False

        # TODO be careful. Could collide with other types!
    def __hash__(self):
        return hash((*self.shape, self.is_solver))


def generate_solver_type(shape):
    return SolverType(shape)
