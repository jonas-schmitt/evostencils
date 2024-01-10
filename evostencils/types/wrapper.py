class TypeWrapper:
    def __init__(self, type_, FAS=False):
        self.type_ = type_
        self.FAS = FAS

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.type_ == other.type_ and self.FAS == other.FAS
        else:
            return False

    def issubtype(self, other):
        if isinstance(other, type(self)):
            return self.type_ == other.type_ and self.FAS == other.FAS
        else:
            return False

    def __hash__(self):
        return hash((type(self), self.type_, self.FAS))
