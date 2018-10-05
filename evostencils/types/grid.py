class GridTypeMetaClass(type):
    def __new__(mcs, class_name, bases, dct):
        return super(GridTypeMetaClass, mcs).__new__(mcs, class_name, bases, dct)

    def __eq__(self, other):
        if hasattr(other, 'size') and hasattr(other, 'grid_type'):
            return self.size == other.size \
                   and self.grid_type == other.grid_type
        else:
            return False

    def __subclasscheck__(self, other):
        if hasattr(other, 'size') and hasattr(other, 'grid_type'):
            is_subclass = True
            if self.size != other.size:
                return False
            if self.grid_type == "grid":
                is_subclass = is_subclass and other.grid_type == "grid"
            elif self.grid_type == "rhs":
                is_subclass = is_subclass and other.grid_type == "rhs"
            elif self.grid_type == "residual":
                is_subclass = is_subclass and other.grid_type == "residual"
            else:
                return False
            return is_subclass
        else:
            return False

    def __hash__(self):
        return hash((*self.size, self.grid_type))


def generate_grid_type(size):
    return GridTypeMetaClass("GridType", (), {"size": size, "grid_type": "grid"})


def generate_residual_type(size):
    return GridTypeMetaClass("GridType", (), {"size": size, "grid_type": "residual"})


def generate_rhs_type(size):
    return GridTypeMetaClass("GridType", (), {"size": size, "grid_type": "rhs"})

