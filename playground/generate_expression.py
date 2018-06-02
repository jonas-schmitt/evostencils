from deap import base, creator, gp, tools
from sympy import *
from evostencils.matrix import *

grid = (2,)

x = generate_vector_on_grid('x', grid)
b = generate_vector_on_grid('b', grid)
A = generate_matrix_on_grid('A', grid)
I = Identity(x.shape[0])
D = get_diagonal(A)
L = get_lower_triangle(A)
U = get_upper_triangle(A)

MatrixType = generate_matrix_type(A.shape)
VectorType = generate_matrix_type(x.shape)
DiagonalMatrixType = generate_matrix_type(A.shape, diag=True, lower_triangle=False, upper_triangle=False)
StrictlyLowerTriangularMatrixType = generate_matrix_type(A.shape, diag=False, lower_triangle=True, upper_triangle=False)
StrictlyUpperTriangularMatrixType = generate_matrix_type(A.shape, diag=False, lower_triangle=False, upper_triangle=True)
LowerTriangularMatrixType = generate_matrix_type(A.shape, diag=True, lower_triangle=True, upper_triangle=False)
UpperTriangularMatrixType = generate_matrix_type(A.shape, diag=True, lower_triangle=False, upper_triangle=True)

pset = gp.PrimitiveSetTyped("main", [], VectorType)

# Define Terminals
pset.addTerminal(x, VectorType, name='x')
pset.addTerminal(b, VectorType, name='b')
pset.addTerminal(A, MatrixType, name='A')
pset.addTerminal(D, DiagonalMatrixType, name='D')
pset.addTerminal(L, StrictlyLowerTriangularMatrixType, name='L')
pset.addTerminal(U, StrictlyUpperTriangularMatrixType, name='U')
pset.addTerminal(D.I, DiagonalMatrixType, name='D.I')
pset.addTerminal(I, DiagonalMatrixType, name='I')


# Define Operators
pset.addPrimitive(MatrixExpr.__mul__, [MatrixType, MatrixType], MatrixType, name='mul')
pset.addPrimitive(MatrixExpr.__mul__, [MatrixType, VectorType], VectorType, name='mul')
pset.addPrimitive(MatrixExpr.__mul__, [DiagonalMatrixType, VectorType], VectorType, name='mul')
pset.addPrimitive(MatrixExpr.__mul__, [DiagonalMatrixType, DiagonalMatrixType], DiagonalMatrixType, name='mul')
pset.addPrimitive(MatrixExpr.__mul__, [DiagonalMatrixType, MatrixType], MatrixType, name='mul')
pset.addPrimitive(MatrixExpr.__mul__, [MatrixType, DiagonalMatrixType], MatrixType, name='mul')
# TODO include operations on remaining matrix types

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
