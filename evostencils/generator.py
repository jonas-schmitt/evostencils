import sympy as sp
from sympy import MatrixSymbol
from deap import gp, creator, base, tools
from evostencils import matrixtypes as mt
import operator
import itertools


class ExpressionGenerator:

    def __init__(self, coefficients: MatrixSymbol, unknowns: MatrixSymbol, rhs: MatrixSymbol):
        self._coefficients = coefficients
        self._unknowns = unknowns
        self._rhs = rhs
        self._symbols = []
        self._types = []
        self._symbol_types = {self._coefficients: mt.generate_matrix_type(self._coefficients.shape)}
        self._primitive_set = gp.PrimitiveSetTyped("main", [], mt.generate_matrix_type(self._coefficients.shape))
        self._init_terminals()
        self._init_operators()
        self._init_creator()
        self._init_toolbox()

    def _init_terminals(self):
        A = self.get_coefficient_matrix
        self.add_terminal(A, self.get_symbol_type(A))
        identity_matrix = sp.Identity(self.get_vector_of_unknowns.shape[0])
        self.add_terminal(identity_matrix, mt.generate_diagonal_matrix_type(A.shape))
        symbols = [mt.get_diagonal(A),
                   mt.get_lower_triangle(A),
                   mt.get_upper_triangle(A)]
        self.add_terminal_list(symbols)

    def _init_operators(self):
        A = self.get_coefficient_matrix
        GeneralMatrixType = mt.generate_matrix_type(A.shape)
        DiagonalMatrixType = mt.generate_diagonal_matrix_type(mt.get_diagonal(self.get_coefficient_matrix))
        StrictlyLowerTriangularMatrixType = \
            mt.generate_strictly_lower_triangular_matrix_type(mt.get_lower_triangle(self.get_coefficient_matrix))
        StrictlyUpperTriangularMatrixType = \
            mt.generate_strictly_upper_triangular_matrix_type(mt.get_upper_triangle(self.get_coefficient_matrix))

        LowerTriangularMatrixType = \
            mt.generate_lower_triangular_matrix_type(mt.get_lower_triangle(self.get_coefficient_matrix))
        UpperTriangularMatrixType = \
            mt.generate_upper_triangular_matrix_type(mt.get_upper_triangle(self.get_coefficient_matrix))

        matrix_addition = operator.add
        operator_name = 'add'
        for argument_types in itertools.product(self.get_matrix_types, repeat=2):
            if argument_types[0] == argument_types[1]:
                self.add_operator(matrix_addition, argument_types, argument_types[0], operator_name)
            elif (argument_types[0] == DiagonalMatrixType and (argument_types[1] == StrictlyLowerTriangularMatrixType or argument_types[1] == LowerTriangularMatrixType)) \
                    or (argument_types[1] == DiagonalMatrixType and (argument_types[0] == StrictlyLowerTriangularMatrixType or argument_types[0] == LowerTriangularMatrixType)):
                self.add_operator(matrix_addition, argument_types, LowerTriangularMatrixType, operator_name)
            elif (argument_types[0] == DiagonalMatrixType and (argument_types[1] == StrictlyUpperTriangularMatrixType or argument_types[1] == UpperTriangularMatrixType)) \
                    or (argument_types[1] == DiagonalMatrixType and (argument_types[0] == StrictlyUpperTriangularMatrixType or argument_types[0] == UpperTriangularMatrixType)):
                self.add_operator(matrix_addition, argument_types, UpperTriangularMatrixType, operator_name)
            else:
                self.add_operator(matrix_addition, argument_types, GeneralMatrixType, operator_name)

    @staticmethod
    def _init_creator():
        creator.create("Individual", gp.PrimitiveTree)

    def _init_toolbox(self):
        self._toolbox = base.Toolbox()
        self._toolbox.register("expr", gp.genHalfAndHalf, pset=self._primitive_set, min_=1, max_=3)
        self._toolbox.register("individual", tools.initIterate, creator.Individual, self._toolbox.expr)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)

    def set_matrix_type(self, symbol, matrix_type):
        self._symbol_types[symbol] = matrix_type

    def get_symbol_type(self, symbol):
        return self._symbol_types[symbol]

    @property
    def get_coefficient_matrix(self) -> MatrixSymbol:
        return self._coefficients

    @property
    def get_vector_of_unknowns(self) -> MatrixSymbol:
        return self._unknowns

    @property
    def get_right_hand_side_vector(self) -> MatrixSymbol:
        return self._rhs

    @property
    def get_symbols(self) -> list:
        return self._symbols

    @property
    def get_matrix_types(self) -> list:
        return self._types

    def add_terminal(self, symbol, matrix_type):
        self._symbols.append(symbol)
        self._types.append(matrix_type)
        self._symbol_types[symbol] = matrix_type
        self._primitive_set.addTerminal(symbol, matrix_type, name=str(symbol))

    def add_terminal_list(self, symbols: list, matrix_types:list=None):
        assert not matrix_types or len(symbols) == len(matrix_types), \
            "Either no or the appropriate number of types must be provided"
        if not matrix_types:
            for symbol in symbols:
                if isinstance(symbol, mt.DiagonalMatrixSymbol):
                    self.add_terminal(symbol, mt.generate_diagonal_matrix_type(symbol.shape))
                elif isinstance(symbol, mt.LowerTriangularMatrixSymbol):
                    self.add_terminal(symbol, mt.generate_strictly_lower_triangular_matrix_type(symbol.shape))
                elif isinstance(symbol, mt.UpperTriangularMatrixSymbol):
                    self.add_terminal(symbol, mt.generate_strictly_upper_triangular_matrix_type(symbol.shape))
                else:
                    self.add_terminal(symbol, mt.generate_matrix_type(symbol.shape))
        else:
            for (symbol, matrix_type) in zip(symbols, matrix_types):
                self.add_terminal(symbol, matrix_type)

    def add_operator(self, primitive, argument_types, result_type, name: str):
        self._primitive_set.addPrimitive(primitive, argument_types, result_type, name)

    def generate_individual(self):
        return self._toolbox.individual()

    def compile_expression(self, expression):
        return gp.compile(expression, self._primitive_set)
