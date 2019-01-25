class GridType:
    def __init__(self, size_, type_):
        self.size = size_
        self.type = type_

    def __eq__(self, other):
        if hasattr(other, 'size') and hasattr(other, 'type'):
            return self.size == other.size and self.type == other.type
        else:
            return False

    def issubtype(self, other):
        if hasattr(other, 'size') and hasattr(other, 'type'):
            is_subtype = True
            if self.size != other.size:
                return False
            if self.type == "grid":
                is_subtype = is_subtype and other.type == "grid"
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
        return hash((*self.size, self.type))


def generate_grid_type(size):
    return GridType(size, 'grid')


def generate_correction_type(size):
    return GridType(size, 'correction')


def generate_rhs_type(size):
    return GridType(size, 'rhs')


