class LevelControlType:
    def __init__(self, finished):
        self.finished = finished

    def __eq__(self, other):
        if hasattr(other, 'finished'):
            return self.finished == other.finished
        else:
            return False

    def issubtype(self, other):
        if hasattr(other, 'finished'):
            return self.finished == other.finished
        else:
            return False

    def __hash__(self):
        return hash(self.finished)


def generate_finished_type():
    return LevelControlType(True)


def generate_not_finished_type():
    return LevelControlType(False)

