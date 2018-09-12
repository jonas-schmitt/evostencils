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


def convert_constant_stencils(func):
    def wrapper(*args, **kwargs):
        def to_periodic_stencil(constant_stencil: constant.Stencil):
            def recursive_descent(dimension):
                if dimension == 1:
                    return constant_stencil,
                else:
                    return recursive_descent(dimension - 1),
            return Stencil(recursive_descent(constant_stencil.dimension), constant_stencil.dimension)

        converted_args = []
        for arg in args:
            if isinstance(arg, constant.Stencil):
                converted_args.append(to_periodic_stencil(arg))
            else:
                converted_args.append(arg)
        converted_kwargs = {}
        for key, arg in kwargs.items():
            if isinstance(arg, constant.Stencil):
                converted_kwargs[key] = to_periodic_stencil(arg)
            else:
                converted_kwargs[key] = arg
        return func(*converted_args, **converted_kwargs)
    return wrapper


@convert_constant_stencils
def map_stencil_with_index(stencil, f):
    if stencil is None:
        return stencil

    def recursive_descent(array, dimension, index):
        if dimension == 1:
            return tuple(f(element, index + (i,)) for i, element in enumerate(array))
        else:
            return tuple(recursive_descent(element, dimension - 1, index + (i,)) for i, element in enumerate(array))

    result = recursive_descent(stencil.constant_stencils, stencil.dimension, ())
    return Stencil(result, stencil.dimension)


@convert_constant_stencils
def map_stencil(stencil, f):
    if stencil is None:
        return stencil

    def recursive_descent(array, dimension):
        if dimension == 1:
            return tuple(f(constant_stencil) for constant_stencil in array)
        else:
            return tuple(recursive_descent(element, dimension - 1) for element in array)

    result = recursive_descent(stencil.constant_stencils, stencil.dimension)
    return Stencil(result, stencil.dimension)


def diagonal(stencil):
    return map_stencil(stencil, constant.diagonal)


def lower(stencil):
    return map_stencil(stencil, constant.lower)


def upper(stencil):
    return map_stencil(stencil, constant.upper)


def transpose(stencil):
    return map_stencil(stencil, constant.transpose)


@convert_constant_stencils
def combine(stencil1, stencil2, f):
    if stencil1 is None or stencil2 is None:
        return None
    assert stencil1.dimension == stencil2.dimension, "Dimensions must match"
    dim = stencil1.dimension

    def recursive_descent(array1, array2, dimension):
        max_period = max(len(array1), len(array2))
        if dimension == 1:
            return tuple(f(array1[i % len(array1)], array2[i % len(array2)]) for i in range(max_period))
        else:
            return tuple(recursive_descent(array1[i % len(array1)], array2[i % len(array2)], dimension - 1) for i in range(max_period))

    result = recursive_descent(stencil1.constant_stencils, stencil2.constant_stencils, dim)
    return Stencil(result, dim)


def add(stencil1, stencil2):
    return combine(stencil1, stencil2, constant.add)


def sub(stencil1, stencil2):
    return combine(stencil1, stencil2, constant.sub)


def mul(stencil1, stencil2):
    return combine(stencil1, stencil2, constant.mul)


def scale(factor, stencil):
    return map_stencil(stencil, lambda s: constant.scale(factor, s))


def create_empty_multidimensional_array(shape):
    assert len(shape) >= 1, "The dimension must be greater or equal 1"

    def recursive_descent(dimension):
        index = -dimension
        if dimension == 1:
            return tuple(None for _ in range(shape[index]))
        else:
            return tuple(recursive_descent(dimension - 1) for _ in range(shape[index]))
    result = recursive_descent(len(shape))
    return result


def block_diagonal(stencil: constant.Stencil, block_size):
    stencils = create_empty_multidimensional_array(block_size)
    periodic_stencil = Stencil(stencils, stencil.dimension)

    def f(_, index):
        def predicate(offset, _):
            tmp = tuple(o + index[i] for i, o in enumerate(offset))
            for i in range(len(tmp)):
                if tmp[i] < 0 or tmp[i] >= block_size[i]:
                    return False
            return True
        return constant.filter_stencil(stencil, predicate)
    return map_stencil_with_index(periodic_stencil, f)



