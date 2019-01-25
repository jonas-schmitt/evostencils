class TypeList:
    def __init__(self, types):
        self.types = types

    def __eq__(self, other):
        if hasattr(other, 'types'):
            return all(type1 == type2 for type1, type2 in zip(self.types, other.types))
        else:
            return False

    def issubtype(self, other):
        if hasattr(other, 'types'):
            return all(type1.issubtype(type2) for type1, type2 in zip(self.types, other.types))
        else:
            return False

    def __hash__(self):
        return hash(tuple(hash(type_) for type_ in self.types))


def generate_type_list(*types):
    return TypeList(types)
