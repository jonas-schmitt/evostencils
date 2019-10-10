import random
from inspect import isclass
import deap.gp


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
        if condition(height, depth):
            if terminals_available:
                nodes = pset.terminals[type_]
                is_primitive = False
            elif primitives_available:
                nodes = pset.primitives[type_]
            else:
                raise RuntimeError(f"Neither terminal nor primitive available for {type_}")
        else:
            if primitives_available:
                nodes = pset.primitives[type_]
            elif terminals_available:
                nodes = pset.terminals[type_]
                is_primitive = False
            else:
                raise RuntimeError(f"Neither terminal nor primitive available for {type_}")
        choice = random.choice(nodes)
        if is_primitive:
            for arg in reversed(choice.args):
                stack.append((depth + 1, arg))
        else:
            if isclass(choice):
                choice = choice()
        expression.append(choice)
    return expression


def genGrow(pset, min_height, max_height, type_=None):
    def condition(height, depth):
        return depth >= height or \
           (depth >= min_height and random.random() < pset.terminalRatio)
    return generate(pset, min_height, max_height, condition, type_)


class PrimitiveSetTyped(deap.gp.PrimitiveSetTyped):

    def _add(self, prim):
        def addType(dict_, ret_type):
            if ret_type not in dict_:
                new_list = []
                for type_, list_ in dict_.items():
                    if ret_type.issubtype(type_):
                        for item in list_:
                            if item not in new_list:
                                new_list.append(item)
                dict_[ret_type] = new_list

        addType(self.primitives, prim.ret)
        addType(self.terminals, prim.ret)

        self.mapping[prim.name] = prim
        if isinstance(prim, deap.gp.Primitive):
            for type_ in prim.args:
                addType(self.primitives, type_)
                addType(self.terminals, type_)
            dict_ = self.primitives
        else:
            dict_ = self.terminals

        for type_ in dict_:
            if type_.issubtype(prim.ret):
                dict_[type_].append(prim)
