class Type:
    def __init__(self, identifier, guard=False):
        self.identifier = identifier
        self.guard = guard

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.identifier == other.identifier and self.guard == other.guard
        else:
            return False

    def __hash__(self):
        return hash((self.identifier, self.guard))
