import abc
import evostencils.stencils.constant as constant
import evostencils.stencils.multiple as multiple


class Partitioning(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def generate(_):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass


class Single:
    @staticmethod
    def generate(stencil, grid):
        if stencil is None:
            return [None]
        else:
            return [constant.get_unit_stencil(grid)]

    @staticmethod
    def get_name():
        return "single"

    def __repr__(self):
        return 'Single()'


class RedBlack:
    @staticmethod
    def generate(stencil, grid):
        if stencil is None:
            return [None]
        else:
            return multiple.red_black_partitioning(stencil, grid)

    @staticmethod
    def get_name():
        return "red_black"

    def __repr__(self):
        return 'RedBlack()'
