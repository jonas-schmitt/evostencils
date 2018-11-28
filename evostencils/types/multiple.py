class TypeListMetaClass(type):
    def __new__(mcs, class_name, bases, dct):
        return super(TypeListMetaClass, mcs).__new__(mcs, class_name, bases, dct)

    def __eq__(self, other):
        if hasattr(other, 'types'):
            return all((type1 == type2 for type1, type2 in zip(self.types, other.types)))
        else:
            return False

    def __subclasscheck__(self, other):
        if hasattr(other, 'types'):
            return all(issubclass(type1, type2) for type1, type2 in zip(self.types, other.types))
        else:
            return False

    def __hash__(self):
        return hash(tuple(self.types))


def generate_type_list(*types):
    return TypeListMetaClass("TypeList", (), {"types": types})


def generate_new_type(name):
    return type(name, (), {})