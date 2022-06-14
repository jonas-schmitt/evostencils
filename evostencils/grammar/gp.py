import random
from inspect import isclass
from deap import gp
from operator import attrgetter

def generate(pset, min_height, max_height, condition, return_type=None, subtree=None):
    if return_type is None:
        type_ = pset.ret
    else:
        type_ = return_type
    expression = []
    height = random.randint(min_height, max_height)
    stack = [(0, type_)]
    max_depth = 0
    subtree_inserted = False
    if subtree is None:
        subtree_inserted = True
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if not subtree_inserted and type_ == return_type and len(expression) > 0:
            expression.extend(subtree)
            subtree_inserted = True
            continue
        max_depth = max(max_depth, depth)
        terminals_available = len(pset.terminals[type_]) > 0
        if condition(height, depth):
            nodes = pset.terminals[type_] + pset.primitives[type_]
        else:
            if terminals_available:
                nodes = pset.terminals[type_]
            else:
                nodes = pset.primitives[type_]
        if len(nodes) == 0:
            raise RuntimeError(f"Neither terminal nor primitive available for {type_}")
        choice = random.choice(nodes)
        if choice.arity > 0:
            for arg in reversed(choice.args):
                stack.append((depth + 1, arg))
        else:
            if isclass(choice):
                choice = choice()
        expression.append(choice)
    return expression


def genGrow(pset, min_height, max_height, type_=None, size_limit=150):
    def condition(height, depth):
        return depth < height
    result = generate(pset, min_height, max_height, condition, type_)
    while len(result) > size_limit: # Include size limit for individual as well
        result = generate(pset, min_height, max_height, condition, type_)
    return result


class PrimitiveSetTyped(gp.PrimitiveSetTyped):
    def _add(self, prim):
        def addType(dict_, ret_type):
            if ret_type not in dict_:
                new_list = []
                for type_, list_ in dict_.items():
                    if ret_type == type_:
                        for item in list_:
                            if item not in new_list:
                                new_list.append(item)
                dict_[ret_type] = new_list

        addType(self.primitives, prim.ret)
        addType(self.terminals, prim.ret)

        self.mapping[prim.name] = prim
        if isinstance(prim, gp.Primitive):
            for type_ in prim.args:
                addType(self.primitives, type_)
                addType(self.terminals, type_)
            dict_ = self.primitives
        else:
            dict_ = self.terminals

        for type_ in dict_:
            if type_ == prim.ret:
                dict_[type_].append(prim)


def mutNodeReplacement(individual, pset):
    """Replaces a randomly chosen primitive from *individual* by a randomly
    chosen primitive with the same number of arguments from the :attr:`pset`
    attribute of the individual.
    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    if len(individual) < 2:
        return individual,

    while True:
        index = random.randrange(1, len(individual))
        node = individual[index]

        if node.arity == 0:  # Terminal
            term = random.choice(pset.terminals[node.ret])
            if isclass(term):
                term = term()
            individual[index] = term
            return individual,
        else:  # Primitive
            prims = [p for p in pset.primitives[node.ret] if p.args == node.args]
            if len(prims) > 1:
                individual[index] = random.choice(prims)
                return individual,


def mutate_subtree(individual, min_height, max_height, pset):
    index = random.randrange(len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)

    def condition(height, depth):
        return depth < height
    if random.random() < 0.5:
        subtree = individual[slice_]
    else:
        subtree = None
    new_subtree = generate(pset, min_height, max_height, condition, node.ret, subtree)
    individual[slice_] = new_subtree
    return individual,


def select_unique_best(individuals, k, fit_attr="fitness"):
    dictionary = {}
    for i, ind in enumerate(individuals):
        key = str(ind)
        if key not in dictionary:
            dictionary[key] = i
    unique_indices = dictionary.items()
    unique_individuals = [individuals[i] for _, i in unique_indices]
    return sorted(unique_individuals, key=attrgetter(fit_attr), reverse=True)[:k]
