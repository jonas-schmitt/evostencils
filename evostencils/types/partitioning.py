class PartitioningType:
    def __init__(self, partitioning):
        self.partitioning = partitioning

    def __eq__(self, other):
        if hasattr(other, 'partitioning'):
            return self.partitioning == other.partitioning
        else:
            return False

    def issubtype(self, other):
        if hasattr(other, 'partitioning'):
            is_subtype = True
            if self.partitioning == "any":
                return True
            elif self.partitioning == "single":
                is_subtype = is_subtype and other.partitioning == "single"
            elif self.partitioning == "red_black":
                is_subtype = is_subtype and other.partitioning == "red_black"
            else:
                return False
            return is_subtype
        else:
            return False

    def __hash__(self):
        return hash(self.partitioning)

def generate_any_partitioning_type():
    return PartitioningType('any')


def generate_red_black_partitioning_type():
    return PartitioningType('red_black')


def generate_single_partitioning_type():
    return PartitioningType('single')
