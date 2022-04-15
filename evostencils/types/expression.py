class ExpressionType:
    def __init__(self, size_, type_):
        self.size = size_
        self.type = type_

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.size == other.size and self.type == other.type
        else:
            return False

    def issubtype(self, other):
        if isinstance(other, type(self)):
            is_subtype = True
            if self.size != other.size:
                return False
            if self.type == "approximation":
                is_subtype = is_subtype and other.type == "approximation"
            elif self.type == "rhs":
                is_subtype = is_subtype and other.type == "rhs"
            elif self.type == "correction":
                is_subtype = is_subtype and other.type == "correction"
            else:
                return False
            return is_subtype
        else:
            return False

    def __hash__(self):
        return hash((type(self), *self.size, self.type))


def generate_approximation_type(size):
    return ExpressionType(size, 'approximation')


def generate_correction_type(size):
    return ExpressionType(size, 'correction')


def generate_rhs_type(size):
    return ExpressionType(size, 'rhs')


