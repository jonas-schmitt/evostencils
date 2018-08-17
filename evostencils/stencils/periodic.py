import evostencils.stencils.constant as constant

class Stencil:
    def __init__(self, entries):
        assert len(entries) > 0, "A periodic stencil must have at least one entry"
        self._entries = entries

    @property
    def entries(self):
        return self._entries

    def dimension(self):
        return len(self.entries)

