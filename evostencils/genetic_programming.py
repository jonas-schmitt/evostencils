import random
import sys
from inspect import isclass


def generate(pset, min_height, max_height, condition, type_=None):
    """Generate a Tree as a list of list. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.

    :param pset: Primitive set from which primitives are selected.
    :param min_height: Minimum height of the produced trees.
    :param max_height: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              depending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expression = []
    height = random.randint(min_height, max_height)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        is_primitive = True
        terminals_available = len(pset.terminals[type_]) > 0
        primitives_available = len(pset.primitives[type_]) > 0
        if (condition(height, depth) or not primitives_available) and terminals_available:
            nodes = pset.terminals[type_]
            is_primitive = False
        elif primitives_available:
            nodes = pset.primitives[type_]
        else:
            raise RuntimeError(f"Neither terminal nor primitive available for {type_}")
        node = random.choice(nodes)
        if is_primitive:
            for arg in reversed(node.args):
                stack.append((depth + 1, arg))
        else:
            if isclass(node):
                node = node()
        expression.append(node)
    return expression


def generate_tree_with_minimum_height(pset, min_height, max_height, type_=None):
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth >= height
    return generate(pset, min_height, max_height, condition, type_)

