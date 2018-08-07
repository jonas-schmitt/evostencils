class Stencil:
    def __init__(self, entries):
        self._entries = tuple(entries)

    @property
    def entries(self):
        return self._entries

    @property
    def dimension(self):
        return len(self.entries[0])

    @property
    def number_of_entries(self):
        return len(self.entries)

    def __repr__(self):
        return f'Stencil({repr(self.entries)})'


def lexicographical_less(a, b):
    for p, q in zip(a, b):
        if p != q:
            # the first non-equal entry determines the result
            return p < q
    # all entries are the same
    return False


def map_stencil(stencil, f):
    if stencil is None:
        return stencil
    result = []
    for offset, value in stencil.entries:
        result.append(f(offset, value))
    return Stencil(result)


def filter_stencil(stencil, predicate):

    if stencil is None:
        return stencil
    result = []
    for offset, value in stencil.entries:
        if predicate(offset, value):
            result.append((offset, value))
    return Stencil(result)


def combine(stencil1, stencil2, f):
    if stencil1 is None or stencil2 is None:
        return None
    new_entries = list(stencil1.entries)
    for entry2 in stencil2.entries:
        added = False
        for i, new_entry in enumerate(new_entries):
            if new_entry[0] == entry2[0]:
                new_entries[i] = (entry2[0], f(new_entry[1], entry2[1]))
                added = True
                break
        if not added:
            new_entries.append(entry2)
    return Stencil(new_entries)


def diagonal(stencil):
    import numpy as np

    def is_diagonal(offset, _):
        return (np.array(offset, np.int) == 0).all()

    return filter_stencil(stencil, is_diagonal)


def lower(stencil):
    import numpy as np
    zero = np.zeros(stencil.dimension)
    return filter_stencil(stencil, lambda o, v: lexicographical_less(o, zero))


def upper(stencil):
    import numpy as np
    zero = np.zeros(stencil.dimension)
    return filter_stencil(stencil, lambda o, v: lexicographical_less(zero, o))


def transpose(self):
    import numpy as np
    return map_stencil(self, lambda o, v: (-np.array(o), v))


# Note that this is not equivalent to matrix inversion
def inverse(self):
    return map_stencil(self, lambda o, v: (o, 1.0 / v))


def add(stencil1, stencil2):
    return combine(stencil1, stencil2, lambda x, y: x + y)


def sub(stencil1, stencil2):
    return combine(stencil1, stencil2, lambda x, y: x - y)


def scale(factor, stencil):
    return map_stencil(stencil, lambda offset, value: (offset, factor * value))


def mul(stencil1, stencil2):
    if stencil1 is None or stencil2 is None:
        return None
    from operator import add as builtin_add
    new_entries = []
    for offset2, value2 in stencil2.entries:
        for offset1, value1 in stencil1.entries:
            added = False
            offset = tuple(map(builtin_add, offset1, offset2))
            value = value1 * value2
            for i, new_entry in enumerate(new_entries):
                if offset == new_entry[0]:
                    new_entries[i] = (offset, new_entry[1] + value)
                    added = True
                    break
            if not added:
                new_entries.append((offset, value))
    return Stencil(new_entries)


def get_unit_stencil(dimension=None) -> Stencil:
    if dimension is None:
        return None
    else:
        offsets = []
        for i in range(dimension):
            offsets.append(0)
        entries = ((tuple(offsets), 1.0),)
        #entries = ((0 for i in range(dimension)), 1.0)
        return Stencil(entries)


def get_null_stencil() -> Stencil:
    return Stencil(entries=())
