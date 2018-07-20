import sympy as sp
import numpy as np
from sympy import BlockMatrix
from deap import gp, creator, base, tools, algorithms
import random
from evostencils import types
from evostencils.expressions import scalar, block
from evostencils.expressions import multigrid
import operator
import functools

def dummy_eval(individual, generator):
    return 0.0


class Optimizer:
    def __init__(self, operator: BlockMatrix, grid: BlockMatrix, rhs: BlockMatrix, evaluate=dummy_eval):
        self._operator = operator
        self._grid = grid
        self._rhs = rhs
        self._diagonal = block.get_diagonal(operator)
        self._symbols = set()
        self._types = set()
        self._symbol_types = {}
        self._symbol_names = {}
        self._primitive_set = gp.PrimitiveSetTyped("main", [], types.generate_matrix_type(self._grid.shape))
        self._init_terminals()
        self._init_operators()
        self._init_creator()
        self._init_toolbox(evaluate)

    def _init_terminals(self):
        A = self._operator
        u = self._grid
        f = self._rhs
        D = self._diagonal

        identity_matrix = sp.Identity(self.grid.shape[0])
        # Add primitives to set
        self.add_terminal(A, types.generate_matrix_type(A.shape), 'A')
        self.add_terminal(u, types.generate_matrix_type(u.shape), 'u')

        self.add_terminal(identity_matrix, types.generate_diagonal_matrix_type(A.shape), 'I')
        self.add_terminal(D, types.generate_diagonal_matrix_type(A.shape), 'A_d')
        self.add_terminal(D.I, types.generate_diagonal_matrix_type(A.shape), 'A_d_inv')
        self.add_terminal(block.get_lower_triangle(A), types.generate_matrix_type(A.shape), 'A_l')
        self.add_terminal(block.get_upper_triangle(A), types.generate_matrix_type(A.shape), 'A_u')

        # Multigrid recipes
        coarsening_factor = 4
        coarse_grid = multigrid.get_coarse_grid(u, coarsening_factor)
        coarse_operator = multigrid.get_coarse_operator(A, coarsening_factor)
        interpolation = multigrid.get_interpolation(u, coarse_grid)
        restriction = multigrid.get_restriction(u, coarse_grid)

        self.add_terminal(sp.ZeroMatrix(*coarse_grid.shape), types.generate_matrix_type(coarse_grid.shape), 'Zero')
        self.add_terminal(coarse_operator.I, types.generate_matrix_type(coarse_operator.shape), 'A_coarse_inv')
        self.add_terminal(interpolation, types.generate_matrix_type(interpolation.shape), 'P')
        self.add_terminal(restriction, types.generate_matrix_type(restriction.shape), 'R')

        self._coarsening_factor = coarsening_factor
        self._coarse_grid = coarse_grid
        self._coarse_operator = coarse_operator
        self._interpolation = interpolation
        self._restriction = restriction



    def _init_operators(self):
        A = self._operator
        u = self._grid
        OperatorType = types.generate_matrix_type(A.shape)
        GridType = types.generate_matrix_type(u.shape)
        DiagonalOperatorType = types.generate_diagonal_matrix_type(block.get_diagonal(A).shape)

        # Add primitives to full set
        self.add_operator(operator.add, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, 'add')
        self.add_operator(operator.add, [OperatorType, OperatorType], OperatorType, 'add')

        self.add_operator(operator.sub, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, 'sub')
        self.add_operator(operator.sub, [OperatorType, OperatorType], OperatorType, 'sub')

        self.add_operator(operator.mul, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, 'mul')
        self.add_operator(operator.mul, [OperatorType, OperatorType], OperatorType, 'mul')

        self.add_operator(sp.MatrixExpr.inverse, [DiagonalOperatorType], DiagonalOperatorType, 'inverse')

        # Correction
        correct = functools.partial(multigrid.correct, operator=self._operator, rhs=self._rhs)
        self.add_operator(correct, [OperatorType, GridType], GridType, 'correct')

        # Multigrid recipes
        InterpolationType = types.generate_matrix_type(self._interpolation.shape)
        RestrictionType = types.generate_matrix_type(self._restriction.shape)
        CoarseOperatorType = types.generate_matrix_type(self._coarse_operator.I.shape)
        CoarseGridType = types.generate_matrix_type(self._coarse_grid.shape)

        # Restriction
        self.add_operator(operator.mul, [RestrictionType, GridType], CoarseGridType, 'restrict')
        # Interpolation
        self.add_operator(operator.mul, [InterpolationType, CoarseGridType], GridType, 'interpolate')

        # Solving on the coarse grid
        self.add_operator(operator.mul, [CoarseOperatorType, CoarseGridType], CoarseGridType, 'mul')

        # Create intergrid operators
        self.add_operator(operator.mul, [CoarseOperatorType, RestrictionType], RestrictionType, 'mul')
        self.add_operator(operator.mul, [InterpolationType, CoarseOperatorType], InterpolationType, 'mul')
        self.add_operator(operator.mul, [InterpolationType, RestrictionType], OperatorType, 'mul')


        # Dummy operations
        def noop(A):
            return A

        self.add_operator(noop, [CoarseOperatorType], CoarseOperatorType, 'noop')
        self.add_operator(noop, [RestrictionType], RestrictionType, 'noop')
        self.add_operator(noop, [InterpolationType], InterpolationType, 'noop')


    @staticmethod
    def _init_creator():
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    def _init_toolbox(self, evaluate):
        self._toolbox = base.Toolbox()
        self._toolbox.register("expression", gp.genHalfAndHalf, pset=self._primitive_set, min_=1, max_=5)
        self._toolbox.register("individual", tools.initIterate, creator.Individual, self._toolbox.expression)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("evaluate", evaluate, generator=self)
        self._toolbox.register("select", tools.selTournament, tournsize=2)
        self._toolbox.register("mate", gp.cxOnePoint)
        self._toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
        self._toolbox.register("mutate", gp.mutUniform, expr=self._toolbox.expr_mut, pset=self._primitive_set)

        self._toolbox.decorate("mate", gp.staticLimit(key=len, max_value=20))
        self._toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=20))

    def set_matrix_type(self, symbol, matrix_type):
        self._symbol_types[symbol] = matrix_type

    def get_symbol_type(self, symbol):
        return self._symbol_types[symbol]

    @property
    def operator(self) -> BlockMatrix:
        return self._operator

    @property
    def grid(self) -> BlockMatrix:
        return self._grid

    @property
    def rhs(self) -> BlockMatrix:
        return self._rhs

    @property
    def get_symbols(self) -> list:
        return self._symbols

    @property
    def get_matrix_types(self) -> list:
        return self._types

    def add_terminal(self, symbol, matrix_type, name=None):
        self._symbols.add(symbol)
        self._types.add(matrix_type)
        self._symbol_types[symbol] = matrix_type
        if name:
            self._symbol_names[symbol] = name
            self._primitive_set.addTerminal(symbol, matrix_type, name=name)
        else:
            self._symbol_names[symbol] = str(symbol)
            self._primitive_set.addTerminal(symbol, matrix_type, name=str(symbol))

    def add_operator(self, primitive, argument_types, result_type, name: str):
        for argument_type in argument_types:
            self._types.add(argument_type)
        self._types.add(result_type)
        self._primitive_set.addPrimitive(primitive, argument_types, result_type, name)

    def generate_individual(self):
        return self._toolbox.individual()

    def compile_expression(self, expression):
        return gp.compile(expression, self._primitive_set)

    def compile_scalar_expression(self, expression):
        return sp.block_collapse(gp.compile(expression, self._primitive_set))

    @staticmethod
    def get_iteration_matrix(expression, grid, rhs):
        from evostencils.expressions.transformations import propagate_zero
        tmp = propagate_zero(expression.subs(rhs, sp.ZeroMatrix(*rhs.shape)))
        return tmp.subs(grid, sp.Identity(grid.shape[0]))

    def simple_gp(self, population, generations, crossover_probability, mutation_probability):
        random.seed()
        pop = self._toolbox.population(n=population)
        hof = tools.HallOfFame(10)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, log = algorithms.eaSimple(pop, self._toolbox, crossover_probability, mutation_probability, generations, stats=stats, halloffame=hof, verbose=True)
        return pop, log, hof

    def harm_gp(self, population, generations, crossover_probability, mutation_probability):
        random.seed()
        pop = self._toolbox.population(n=population)
        hof = tools.HallOfFame(10)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = gp.harm(pop, self._toolbox, crossover_probability, mutation_probability, generations, alpha=0.05, beta=10, gamma=0.25, rho=0.9, stats=mstats,
                           halloffame=hof, verbose=True)
        return pop, log, hof

    def default_optimization(self, population, generations, crossover_probability, mutation_probability):
        return self.harm_gp(population, generations, crossover_probability, mutation_probability)




