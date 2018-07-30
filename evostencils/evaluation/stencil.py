def lex_less(a, b):
    """Lexicographical comparison."""

    for p, q in zip(a, b):
        if p != q:
            # the first non-equal entry determines the result
            return p < q

    # all entries are the same
    return False


class Stencil:
    def __init__(self, entries=None):
        self._entries = sorted(entries, key=lambda entry: entry.first)

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
        result = Stencil()
        for offset, value in self.entries:
            result.append(*f(offset, value))
        return result

    def filter(self, take_pred):
        result = Stencil()
        for offset, value in self.entries:
            if take_pred(offset, value):
                result.append(offset, value)
        return result

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
    new_entries = []
    index = 0
    for entry1 in stencil1.entries:
        while index < len(stencil2.entries):
            entry2 = stencil2.entries[index]
            index += 1
            if entry1[0] == entry2[0]:
                new_entries.append((entry1[0], fun(entry1[1], entry2[2])))
                break
    return Stencil(tuple(new_entries))


def add(stencil1: Stencil, stencil2):
    return combine(stencil1, stencil2, lambda x, y: x + y)


def sub(stencil1: Stencil, stencil2):
    return combine(stencil1, stencil2, lambda x, y: x - y)


def scale(factor, stencil: Stencil):
    #TODO
    pass


def mul(stencil1: Stencil, stencil2):
    #TODO
    pass


