from evostencils.expressions import base


class Operator(base.Entity):

    def __init__(self, name, entries):
        self._name = name
        self._entries = entries
        super().__init__()

    @property
    def entries(self):
        return self._entries


class Approximation(base.Entity):
    pass
