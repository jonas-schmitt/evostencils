class GridTypeMetaClass(type):
    def __new__(mcs, class_name, bases, dct):
        return super(GridTypeMetaClass, mcs).__new__(mcs, class_name, bases, dct)

    def __eq__(self, other):
        if hasattr(other, 'size') and hasattr(other, 'is_residual'):
            return self.size == other.size \
                   and self.is_residual == other.is_residual
        else:
            return False

    def __subclasscheck__(self, other):
        if hasattr(other, 'size') and hasattr(other, 'residual'):
            is_subclass = True
            if self.size != other.size:
                return False
            if not self.is_residual:
                is_subclass = is_subclass and not other.is_residual
            return is_subclass
        else:
            return False

    def __hash__(self):
        return hash((self.size, self.is_residual))


def generate_grid_type(size):
    return GridTypeMetaClass("MatrixType", (), {"size": size, "is_residual": False})


def generate_residual_type(size):
    return GridTypeMetaClass("MatrixType", (), {"size": size, "is_residual": True})

