from deap import base, creator, gp, tools
from sympy import *
from evostencils.matrix import *

grid = (2,)

x = generate_vector_on_grid('x', grid)
b = generate_vector_on_grid('b', grid)
A = generate_matrix_on_grid('A', grid)
I = Identity(x.shape[0])
D = diagonal(A)
L = lower(A)
U = upper(A)
MatrixType = generate_operator_type(A.shape, False)
InvertibleMatrixType = generate_operator_type(A.shape, True)
VectorType = generate_operator_type(x.shape, False)

pset = gp.PrimitiveSetTyped("main", [], VectorType)

# Define Terminals
pset.addTerminal(x, VectorType, name='x')
pset.addTerminal(b, VectorType, name='b')
pset.addTerminal(A, MatrixType, name='A')
pset.addTerminal(D, InvertibleMatrixType, name='D')
pset.addTerminal(L, InvertibleMatrixType, name='L')
pset.addTerminal(U, InvertibleMatrixType, name='U')
#pset.addTerminal(D.I, MatrixType, name='D.I')
#pset.addTerminal(I, MatrixType, name='I')


# Define Operators
pset.addPrimitive(MatrixExpr.__mul__, [MatrixType, MatrixType], MatrixType, name='mul')
pset.addPrimitive(MatrixExpr.__mul__, [MatrixType, VectorType], VectorType, name='mul')
pset.addPrimitive(MatrixExpr.__mul__, [InvertibleMatrixType, VectorType], VectorType, name='mul')
pset.addPrimitive(MatrixExpr.__mul__, [InvertibleMatrixType, InvertibleMatrixType], InvertibleMatrixType, name='mul')
pset.addPrimitive(MatrixExpr.__mul__, [InvertibleMatrixType, MatrixType], MatrixType, name='mul')
pset.addPrimitive(MatrixExpr.__mul__, [MatrixType, InvertibleMatrixType], MatrixType, name='mul')
pset.addPrimitive(MatrixExpr.__add__, [MatrixType, MatrixType], MatrixType, name='add')
pset.addPrimitive(MatrixExpr.__add__, [InvertibleMatrixType, InvertibleMatrixType], InvertibleMatrixType, name='add')
pset.addPrimitive(MatrixExpr.__add__, [InvertibleMatrixType, MatrixType], MatrixType, name='add')
pset.addPrimitive(MatrixExpr.__add__, [MatrixType, InvertibleMatrixType], MatrixType, name='add')
pset.addPrimitive(MatrixExpr.__add__, [VectorType, VectorType], VectorType, name='add')
pset.addPrimitive(transpose, [MatrixType], MatrixType, name='transpose')
pset.addPrimitive(transpose, [InvertibleMatrixType], InvertibleMatrixType, name='transpose')
pset.addPrimitive(diagonal, [MatrixType], InvertibleMatrixType, name='diag')
pset.addPrimitive(MatrixExpr.inverse, [InvertibleMatrixType], InvertibleMatrixType, "inverse")
#pset.addPrimitive(simplify, [InvertibleMatrixType], MatrixType, "None")


creator.create("Individual", gp.PrimitiveTree)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


if __name__ == '__main__':
    pop = toolbox.population(n=1000)
    expr = toolbox.individual()
    print(simplify(gp.compile(expr, pset)))
    nodes, edges, labels = gp.graph(expr)

    ## Graphviz Section ###
    import pygraphviz as pgv

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.png", "png")
