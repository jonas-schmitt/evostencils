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
    def generate(stencil):
        if stencil is None:
            return [None]
        else:
            return [constant.get_unit_stencil(stencil.dimension)]


class RedBlack:
    @staticmethod
    def generate(stencil):
        if stencil is None:
            return [None]
        else:
            return periodic.red_black_partitioning(stencil)

