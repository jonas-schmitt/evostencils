import abc
import evostencils.stencils.constant as constant
import evostencils.stencils.periodic as periodic


class Partitioning(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def generate(_):
        pass


class Single:
    @staticmethod
    def generate(stencil, grid):
        if stencil is None:
            return [None]
        else:
            return [constant.get_unit_stencil(grid)]

    def __repr__(self):
        return 'Single()'


class RedBlack:
    @staticmethod
    def generate(stencil, grid):
        if stencil is None:
            return [None]
        else:
            return periodic.red_black_partitioning(stencil, grid)

    def __repr__(self):
        return 'RedBlack()'

