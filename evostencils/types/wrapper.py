class TypeWrapper:
    def __init__(self, type_):
        self.type_ = type_

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.type_ == other.type_
        else:
            return False

    def issubtype(self, other):
        if isinstance(other, type(self)):
            return self.type_ == other.type_
        else:
            return False

    def __hash__(self):
        return hash((type(self), self.type_))
