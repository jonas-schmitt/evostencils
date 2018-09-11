import evostencils.stencils.constant as constant


class Stencil:
    def __init__(self, constant_stencils, dimension):
        assert len(constant_stencils) > 0, "A periodic stencil must contain at least one constant stencil"
        assert dimension >= 1, "The dimension of the problem must be greater or equal 1"
        self._constant_stencils = tuple(constant_stencils)
        self._dimension = dimension

    @property
    def constant_stencils(self):
        return self._constant_stencils

    @property
    def dimension(self):
        return self._dimension


def convert_constant_to_periodic_stencil(constant_stencil: constant.Stencil):
    def recurse(dimension):
        if dimension == 1:
            return constant_stencil
        else:
            return recurse(dimension-1),
    return recurse(constant_stencil.dimension)


def map_stencil(periodic_stencil: Stencil, f):
    if periodic_stencil is None:
        return periodic_stencil

    def recurse(array, dimension):
        if dimension == 1:
            return tuple(f(constant_stencil) for constant_stencil in array)
        else:
            return tuple(recurse(element, dimension - 1) for element in array)

    result = recurse(periodic_stencil.constant_stencils, periodic_stencil.dimension)
    return Stencil(result, periodic_stencil.dimension)


def diagonal(periodic_stencil: Stencil):
    return map_stencil(periodic_stencil, constant.diagonal)


def lower(periodic_stencil: Stencil):
    return map_stencil(periodic_stencil, constant.lower)


def upper(periodic_stencil: Stencil):
    return map_stencil(periodic_stencil, constant.upper)


def transpose(periodic_stencil: Stencil):
    return map_stencil(periodic_stencil, constant.transpose)


def combine(periodic_stencil1: Stencil, periodic_stencil2: Stencil, f):
    if periodic_stencil1 is None or periodic_stencil2 is None:
        return None
    assert periodic_stencil1.dimension == periodic_stencil2.dimension, "Dimensions must match"
    dim = periodic_stencil1.dimension

    def recurse(array1, array2, dimension):
        max_period = max(len(array1), len(array2))
        if dimension == 1:
            return tuple(f(array1[i % len(array1)], array2[i % len(array2)]) for i in max_period)
        else:
            return tuple(recurse(array1[i % len(array1)], array2[i % len(array2)]) for i in max_period)

    result = recurse(periodic_stencil1.constant_stencils, periodic_stencil2.constant_stencils, dim)
    return Stencil(result, dim)


def add(periodic_stencil1: Stencil, periodic_stencil2: Stencil):
    return combine(periodic_stencil1, periodic_stencil2, constant.add)


def sub(periodic_stencil1: Stencil, periodic_stencil2: Stencil):
    return combine(periodic_stencil1, periodic_stencil2, constant.sub)


def mul(periodic_stencil1: Stencil, periodic_stencil2: Stencil):
    return combine(periodic_stencil1, periodic_stencil2, constant.mul)


def scale(factor, periodic_stencil: Stencil):
    return map_stencil(periodic_stencil, lambda s: constant.scale(factor, s))


def create_ndimensional_array_of_stencils(shape):
    assert len(shape) >= 1, "The dimension must be greater or equal 1"

    def recurse(dimension):
        index = -dimension
        if dimension == 1:
            return tuple(None for _ in shape[index])
        else:
            return tuple(recurse(dimension - 1) for _ in shape[index])
    return recurse(len(shape))
