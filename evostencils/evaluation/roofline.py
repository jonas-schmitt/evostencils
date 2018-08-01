from evostencils.expressions import base, multigrid


class RooflineEvaluator:
    def __init__(self, peak_performance, bandwidth, bytes_per_word):
        self._peak_performance= peak_performance
        self._bandwidth = bandwidth
        self._bytes_per_word = bytes_per_word

    @property
    def peak_performance(self):
        return self._peak_performance

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def bytes_per_word(self):
        return self._bytes_per_word

    def estimate_performance(self, operations, words):
        return min(self.peak_performance, operations / (words * self.bytes_per_word) * self.bandwidth)

    def count_operations(self, expression):
        pass


