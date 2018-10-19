from evostencils.expressions import base, partitioning as part


class Cycle(base.Expression):
    def __init__(self, unknown, rhs, coarse_unknown, coarse_rhs, correction, partitioning=part.Single, weight=1.0, predecessor=None):
        # assert iterate.shape == correction.shape, "Shapes must match"
        # assert iterate.grid.size == correction.grid.size and iterate.grid.step_size == correction.grid.step_size, \
        #    "Grids must match"
        self._unknown = unknown
        self._rhs = rhs
        self._coarse_unknown = coarse_unknown
        self._coarse_rhs = coarse_rhs
        self._correction = correction
        self._partitioning = partitioning
        self._weight = weight
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
    def coarse_unknown(self):
        return self._coarse_unknown

    @property
    def coarse_rhs(self):
        return self._coarse_rhs

    @property
    def correction(self):
        return self._correction

    @property
    def partitioning(self):
        return self._partitioning

    @property
    def weight(self):
        return self._weight

    @staticmethod
    def generate_stencil():
        return None

    def generate_expression(self):
        return base.Addition(self.unknown, base.Scaling(self.weight, self.correction))

    def __repr__(self):
        return f'Cycle({repr(self.unknown)}, {repr(self.rhs)}, {repr(self.coarse_unknown)}, ' \
               f'{repr(self.coarse_rhs)}, {repr(self.correction)}, {repr(self.partitioning)}, {repr(self.weight)}'

    def __str__(self):
        return str(self.generate_expression())

    def apply(self, transform: callable, *args):
        unknown = transform(self.unknown, *args)
        rhs = transform(self.rhs, *args)
        coarse_unknown = transform(self.coarse_unknown, *args)
        coarse_rhs = transform(self.coarse_rhs, *args)
        correction = transform(self.correction, *args)
        return Cycle(unknown, rhs, coarse_unknown, coarse_rhs, correction, self.partitioning, self.weight, self.predecessor)


def cycle(unknown, rhs, coarse_unknown, coarse_rhs, correction, partitioning=part.Single, weight=1.0, predecessor=None):
    return Cycle(unknown, rhs, coarse_unknown, coarse_rhs, correction, partitioning, weight, predecessor)
