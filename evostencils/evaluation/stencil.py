def lex_less(a, b):
    """Lexicographical comparison."""

    for p, q in zip(a, b):
        if p != q:
            # the first non-equal entry determines the result
            return p < q

    # all entries are the same
    return False


class Stencil:
    def __init__(self, entries):
        self._entries = tuple(entries)

    @property
    def entries(self):
        return self._entries

    def __iter__(self):
        for entry in self.entries:
            yield entry

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, p):
        """Return the element at offset p."""
        return self.entries[p]

    def map(self, f):
        result = []
        for offset, value in self.entries:
            result.append(*f(offset, value))
        return Stencil(result)

    def filter(self, take_pred):
        result = []
        for offset, value in self.entries:
            if take_pred(offset, value):
                result.append(offset, value)
        return Stencil(result)

    def dim(self):
        return len(self.entries[0])

    def diagonal(self):
        import numpy as np

        def is_diag(offset, _):
            return (np.array(offset, np.int) == 0).all()

        return self.filter(is_diag)

    def lower(self):
        import numpy as np
        zero = np.zeros(self.dim)

        return self.filter(lambda o, v: lex_less(o, zero))

    def upper(self):
        import numpy as np
        zero = np.zeros(self.dim)

        return self.filter(lambda o, v: lex_less(zero, o))

    def transpose(self):
        import numpy as np
        return self.map(lambda o, v: (-np.array(o), v))

    def conjugate(self):
        return self.map(lambda o, v: (o, v.conjugate()))


def combine(stencil1: Stencil, stencil2: Stencil, fun):
    new_entries = list(stencil1.entries)
    for entry2 in stencil2.entries:
        added = False
        for new_entry in new_entries:
            if new_entry[0] == entry2[0]:
                new_entry[1] += entry2[1]
                added = True
                break
        if not added:
            new_entries.append(entry2)
    return Stencil(new_entries)


def add(stencil1: Stencil, stencil2):
    return combine(stencil1, stencil2, lambda x, y: x + y)


def sub(stencil1: Stencil, stencil2):
    return combine(stencil1, stencil2, lambda x, y: x - y)


def scale(factor, stencil: Stencil):
    return stencil.map(lambda offset, value: (offset, factor * value))


def mul(stencil1: Stencil, stencil2):
    import operator.add
    new_entries = []
    for offset2, value2 in stencil2.entries:
        for offset1, value1 in stencil1.entries:
            added = False
            offset = tuple(map(operator.add, offset1, offset2))
            value = value1 * value2
            for new_entry in new_entries:
                if offset == new_entry[0]:
                    new_entry[1] += value
                    added = True
                    break
            if not added:
                new_entries.append((offset, value))
    return Stencil(new_entries)
