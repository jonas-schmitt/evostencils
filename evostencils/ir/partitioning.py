import abc
import evostencils.stencils.constant as constant
import evostencils.stencils.multiple as multiple


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
            return multiple.red_black_partitioning(stencil, grid)

    def __repr__(self):
        return 'RedBlack()'


class FourWay:
    @staticmethod
    def generate(stencil, grid):
        if stencil is None:
            return [None]
        else:
            raise RuntimeError("Stencil generation for 4-way partitioning not implemented")

    def __repr__(self):
        return 'FourWay()'


class NineWay:
    @staticmethod
    def generate(stencil, grid):
        if stencil is None:
            return [None]
        else:
            raise RuntimeError("Stencil generation for 9-way partitioning not implemented")

    def __repr__(self):
        return 'NineWay()'


class EightWay:
    @staticmethod
    def generate(stencil, grid):
        if stencil is None:
            return [None]
        else:
            raise RuntimeError("Stencil generation for 8-way partitioning not implemented")

    def __repr__(self):
        return 'EightWay()'


class TwentySevenWay:
    @staticmethod
    def generate(stencil, grid):
        if stencil is None:
            return [None]
        else:
            raise RuntimeError("Stencil generation for 27-way partitioning not implemented")

    def __repr__(self):
        return 'TwentySevenWay()'
