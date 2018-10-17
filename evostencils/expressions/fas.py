from evostencils.expressions import base, partitioning as part


class CycleFAS(base.Expression):
    def __init__(self, unknown, rhs, new_unknown, new_rhs, partitioning=part.Single, predecessor=None):
        # assert iterate.shape == correction.shape, "Shapes must match"
        # assert iterate.grid.size == correction.grid.size and iterate.grid.step_size == correction.grid.step_size, \
        #    "Grids must match"
        self._unknown = unknown
        self._rhs = rhs
        self._new_unknown = new_unknown
        self._new_rhs = new_rhs
        self._partitioning = partitioning
        self.predecessor = predecessor

    @property
    def shape(self):
        return self._unknown.shape

    @property
    def grid(self):
        return self.unknown.grid

    @property
    def unknown(self):
        return self._unknown

    @property
    def rhs(self):
        return self._rhs

    @property
    def new_unknown(self):
        return self._new_unknown

    @property
    def new_rhs(self):
        return self._new_rhs

    @property
    def partitioning(self):
        return self._partitioning

    @staticmethod
    def generate_stencil():
        return None

    def __repr__(self):
        return f'Cycle({repr(self.new_unknown)}, {repr(self.unknown)}, {repr(self.partitioning)}, {repr(self.weight)}'

    def __str__(self):
        return str(self.generate_expression())

    def apply(self, transform: callable, *args):
        unknown = transform(self.unknown, *args)
        rhs = transform(self.rhs, *args)
        new_unknown = transform(self.new_unknown, *args)
        new_rhs = transform(self.new_rhs, *args)
        return CycleFAS(unknown, rhs, new_unknown, new_rhs, self.partitioning, self.predecessor)


def cycle_fas(unknown, rhs, new_unknown, new_rhs, partitioning=part.Single, predecessor=None):
    return CycleFAS(unknown, rhs, new_unknown, new_rhs, partitioning, predecessor)
