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


def check_predicate(stencil, predicate):
    if stencil is None:
        return False

    def recursive_descent(array, dimension):
        if dimension == 1:
            return all(map(predicate, array))
        else:
            return all(recursive_descent(element, dimension - 1) for element in array)

    return recursive_descent(stencil.constant_stencils, stencil.dimension)


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
def is_diagonal(stencil):
    def predicate(constant_stencil):
        return all(all(i == 0 for i in offset) for offset, _ in constant_stencil.entries)
    return check_predicate(stencil, predicate)


@convert_constant_stencils
def count_number_of_entries(stencil):
    from itertools import chain

    def recursive_descent(array, dimension):
        if dimension == 1:
            return (element.number_of_entries for element in array if element is not None)
        else:
            result = chain()
            for element in array:
                result = chain(result, recursive_descent(element, dimension - 1))
            return result
    return tuple(recursive_descent(stencil.constant_stencils, stencil.dimension))


@convert_constant_stencils
def get_list_of_entries(stencil):
    from itertools import chain

    def recursive_descent(array, dimension):
        if dimension == 1:
            return (element for element in array if element is not None)
        else:
            result = chain()
            for element in array:
                result = chain(result, recursive_descent(element, dimension - 1))
            return result
    return tuple(recursive_descent(stencil.constant_stencils, stencil.dimension))


@convert_constant_stencils
def determine_maximal_shape(stencil):
    def recursive_descent(array, dimension):
        if dimension == 1:
            return [len(array)]
        else:
            tmp = [recursive_descent(element, dimension - 1) for element in array]
            max_shape = tmp[0]
            for a in tmp:
                for i, b in enumerate(a):
                    if b > max_shape[i]:
                        max_shape[i] = b
            return [len(array)] + max_shape
    return recursive_descent(stencil.constant_stencils, stencil.dimension)


@convert_constant_stencils
def map_stencil(stencil, f):
    return indexed_map_stencil(stencil, lambda s, i: f(s))


@convert_constant_stencils
def indexed_map_stencil(stencil, f):
    if stencil is None:
        return stencil

    def recursive_descent(array, dimension, index):
        if dimension == 1:
            return tuple(f(element, index + (i,)) for i, element in enumerate(array))
        else:
            return tuple(recursive_descent(element, dimension - 1, index + (i,)) for i, element in enumerate(array))

    result = recursive_descent(stencil.constant_stencils, stencil.dimension, ())
    return Stencil(result, stencil.dimension)


def diagonal(stencil):
    return map_stencil(stencil, constant.diagonal)


def lower(stencil):
    return map_stencil(stencil, constant.lower)


def upper(stencil):
    return map_stencil(stencil, constant.upper)


def transpose(stencil):
    return map_stencil(stencil, constant.transpose)


def inverse(stencil):
    return map_stencil(stencil, constant.inverse)


@convert_constant_stencils
def combine(stencil1, stencil2, f):
    return indexed_combine(stencil1, stencil2, lambda s1, s2, i: f(s1, s2))


@convert_constant_stencils
def indexed_combine(stencil1, stencil2, f):
    if stencil1 is None or stencil2 is None:
        return None
    assert stencil1.dimension == stencil2.dimension, "Dimensions must match"
    dim = stencil1.dimension

    def recursive_descent(array1, array2, dimension, index):
        max_period = max(len(array1), len(array2))
        if dimension == 1:
            return tuple(f(array1[i % len(array1)], array2[i % len(array2)], index + (i,)) for i in range(max_period))
        else:
            return tuple(recursive_descent(array1[i % len(array1)], array2[i % len(array2)], dimension - 1, index + (i,)) for i in range(max_period))

    result = recursive_descent(stencil1.constant_stencils, stencil2.constant_stencils, dim, ())
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


def block_diagonal(stencil, block_size):
    assert len(block_size) == stencil.dimension, 'Block size does not match dimension of the problem'
    stencils = create_empty_multidimensional_array(block_size)
    empty_stencil = Stencil(stencils, stencil.dimension)

    def f(constant_stencil, _, index):
        def predicate(offset, _):
            tmp = tuple(o + index[i] for i, o in enumerate(offset))
            for i in range(len(tmp)):
                if tmp[i] < 0 or tmp[i] >= block_size[i]:
                    return False
            return True
        return constant.filter_stencil(constant_stencil, predicate)
    return indexed_combine(stencil, empty_stencil, f)


def red_black_partitioning(stencil, grid):
    if stencil is None:
        return None
    tmp = determine_maximal_shape(stencil)
    shape = tuple(2 * n for n in tmp)
    empty_stencil = Stencil(create_empty_multidimensional_array(shape), stencil.dimension)

    def red(_, index):
        if sum(tuple(i // j for i, j in zip(index, tmp))) % 2 == 0:
            return constant.get_unit_stencil(grid)
        else:
            return constant.get_null_stencil(grid)

    def black(_, index):
        if sum(tuple(i // j for i, j in zip(index, tmp))) % 2 == 0:
            return constant.get_null_stencil(grid)
        else:
            return constant.get_unit_stencil(grid)
    red_filter = indexed_map_stencil(empty_stencil, red)
    black_filter = indexed_map_stencil(empty_stencil, black)
    return red_filter, black_filter

"""
def count_zeros_in_system_of_equations(stencil: Stencil):
    entries = stencil.constant_stencils
    nentries = count_number_of_entries(stencil)
    import functools
    zeros = [0 for _ in range(functools.reduce(lambda x, y: x * y, nentries))]

    def recursive_descent(array, dimension, index):
        if dimension == 1:
            for i, element in enumerate(array):
                tmp = index + (i,)
                for offsets, _ in element.entries:
                   for k, index_sum in enumerate([a + b for a, b in zip(tmp, offsets)]):
                       if index_sum > 0 and index_sum < nentries[k]:

        else:
            for i, element in enumerate(array):
                recursive_descent(element, dimension - 1, index + (i,))
    recursive_descent(entries, stencil.dimension, ())
"""




