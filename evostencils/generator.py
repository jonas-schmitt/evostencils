import sympy as sp
from sympy import MatrixSymbol, BlockMatrix
from deap import gp, creator, base, tools
from evostencils import types
from evostencils.expressions import scalar, block
import operator
import copy
import itertools


class Individual:
    def __init__(self, tree1, tree2):
        self.tree1 = gp.PrimitiveTree(tree1)
        self.tree2 = gp.PrimitiveTree(tree2)


def initIterate(container, generator1, generator2):
    return container(generator1(), generator2())


class SmootherGenerator:

    def __init__(self, A: BlockMatrix, x: BlockMatrix, b: BlockMatrix):
        self._A = A
        self._x = x
        self._b = b
        self._D = block.get_diagonal(A)
        self._symbols = set()
        self._types = set()
        self._symbol_types = {}
        self._symbol_names = {}
        self._invertible_primitive_set = gp.PrimitiveSetTyped("main", [], types.generate_diagonal_matrix_type(self._A.shape))
        self._primitive_set = gp.PrimitiveSetTyped("main", [], types.generate_matrix_type(self._A.shape))
        self._init_terminals()
        self._init_operators()
        self._init_creator()
        self._init_toolbox()

    def _init_terminals(self):
        D = self._D
        A = self._A
        identity_matrix = sp.Identity(self.x.shape[0])
        # Add primitives to full set
        self.add_terminal(self._primitive_set, A, types.generate_matrix_type(A.shape), 'A')
        self.add_terminal(self._primitive_set, identity_matrix, types.generate_diagonal_matrix_type(A.shape), 'I')
        self.add_terminal(self._primitive_set, D, types.generate_diagonal_matrix_type(A.shape), 'A_d')
        self.add_terminal(self._primitive_set, block.get_lower_triangle(A), types.generate_strictly_lower_triangular_matrix_type(A.shape), 'A_l')
        self.add_terminal(self._primitive_set, block.get_upper_triangle(A), types.generate_strictly_upper_triangular_matrix_type(A.shape), 'A_u')

        # Add primitives to invertible set
        self.add_terminal(self._invertible_primitive_set, identity_matrix, types.generate_diagonal_matrix_type(A.shape), 'I')
        self.add_terminal(self._invertible_primitive_set, D, types.generate_diagonal_matrix_type(A.shape), 'A_d')



    def _init_operators(self):
        A = self._A
        GeneralMatrixType = types.generate_matrix_type(A.shape)
        DiagonalMatrixType = types.generate_diagonal_matrix_type(block.get_diagonal(A).shape)

        StrictlyLowerTriangularMatrixType = \
            types.generate_strictly_lower_triangular_matrix_type(block.get_lower_triangle(A).shape)
        StrictlyUpperTriangularMatrixType = \
            types.generate_strictly_upper_triangular_matrix_type(block.get_upper_triangle(A).shape)
        LowerTriangularMatrixType = \
            types.generate_lower_triangular_matrix_type(block.get_lower_triangle(A).shape)
        UpperTriangularMatrixType = \
            types.generate_upper_triangular_matrix_type(block.get_upper_triangle(A).shape)

        # Add primitives to full set
        self.add_operator(self._primitive_set, operator.add, [DiagonalMatrixType, DiagonalMatrixType], DiagonalMatrixType, 'add')
        self.add_operator(self._primitive_set, operator.add, [GeneralMatrixType, GeneralMatrixType], GeneralMatrixType, 'add')

        self.add_operator(self._primitive_set, operator.sub, [DiagonalMatrixType, DiagonalMatrixType], DiagonalMatrixType, 'sub')
        self.add_operator(self._primitive_set, operator.sub, [GeneralMatrixType, GeneralMatrixType], GeneralMatrixType, 'sub')

        self.add_operator(self._primitive_set, operator.mul, [DiagonalMatrixType, DiagonalMatrixType], DiagonalMatrixType, 'mul')
        self.add_operator(self._primitive_set, operator.mul, [GeneralMatrixType, GeneralMatrixType], GeneralMatrixType, 'mul')

        self.add_operator(self._primitive_set, sp.MatrixExpr.transpose, [GeneralMatrixType], GeneralMatrixType, 'transpose')

        self.add_operator(self._primitive_set, sp.MatrixExpr.inverse, [DiagonalMatrixType], DiagonalMatrixType, 'inverse')

        # Add primitives to invertible set
        self.add_operator(self._invertible_primitive_set, operator.add, [DiagonalMatrixType, DiagonalMatrixType], DiagonalMatrixType, 'add')

        self.add_operator(self._invertible_primitive_set, operator.sub, [DiagonalMatrixType, DiagonalMatrixType], DiagonalMatrixType, 'sub')

        self.add_operator(self._invertible_primitive_set, operator.mul, [DiagonalMatrixType, DiagonalMatrixType], DiagonalMatrixType, 'mul')

        self.add_operator(self._invertible_primitive_set, sp.MatrixExpr.inverse, [DiagonalMatrixType], DiagonalMatrixType, 'inverse')

    @staticmethod
    def _init_creator():
        creator.create("Individual", Individual)

    def _init_toolbox(self):
        self._toolbox = base.Toolbox()
        self._toolbox.register("invertible_expression", gp.genHalfAndHalf, pset=self._invertible_primitive_set, min_=1, max_=4)
        self._toolbox.register("expression", gp.genHalfAndHalf, pset=self._primitive_set, min_=2, max_=4)
        self._toolbox.register("individual", initIterate, Individual, self._toolbox.invertible_expression, self._toolbox.expression)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)

    def set_matrix_type(self, symbol, matrix_type):
        self._symbol_types[symbol] = matrix_type

    def get_symbol_type(self, symbol):
        return self._symbol_types[symbol]

    @property
    def A(self) -> BlockMatrix:
        return self._A

    @property
    def x(self) -> BlockMatrix:
        return self._x

    @property
    def b(self) -> BlockMatrix:
        return self._b

    @property
    def get_symbols(self) -> list:
        return self._symbols

    @property
    def get_matrix_types(self) -> list:
        return self._types

    def add_terminal(self, pset, symbol, matrix_type, name=None):
        self._symbols.add(symbol)
        self._types.add(matrix_type)
        self._symbol_types[symbol] = matrix_type
        if name:
            self._symbol_names[symbol] = name
            pset.addTerminal(symbol, matrix_type, name=name)
        else:
            self._symbol_names[symbol] = str(symbol)
            pset.addTerminal(symbol, matrix_type, name=str(symbol))

    def add_operator(self, pset, primitive, argument_types, result_type, name: str):
        for argument_type in argument_types:
            self._types.add(argument_type)
        self._types.add(result_type)
        pset.addPrimitive(primitive, argument_types, result_type, name)

    def generate_individual(self):
        return self._toolbox.individual()

    def compile_expression(self, expression):
        return gp.compile(expression, self._primitive_set)

    def compile_scalar_individual(self, individual: Individual) -> sp.MatrixExpr:
        M = sp.block_collapse(self.compile_expression(individual.tree1))
        N = sp.block_collapse(self.compile_expression(individual.tree2))
        M_inv = sp.MatrixExpr.inverse(M)
        x = sp.block_collapse(self._x)
        b = sp.block_collapse(self._b)
        return M_inv * N * x + M_inv * b
