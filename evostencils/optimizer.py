import numpy as np
import deap.base
from deap import gp, creator, tools, algorithms
import random
from evostencils import types
import evostencils.expressions.base as base
import evostencils.expressions.multigrid as multigrid
import evostencils.expressions.transformations as transformations
import operator
import functools
from evostencils.stencils import Stencil
from evostencils.weight_optimizer import WeightOptimizer


class AST(gp.PrimitiveTree):
    def __init__(self, content):
        self._weights = None
        super(AST, self).__init__(content)

    @property
    def weights(self):
        return self._weights

    def set_weights(self, weights):
        self._weights = weights


class Optimizer:
    def __init__(self, op: base.Operator, grid: base.Grid, rhs: base.Grid, dimension, coarsening_factor,
                 convergence_evaluator=None, performance_evaluator=None, epsilon=1e-9, infinity=1e10):
        assert convergence_evaluator is not None, "At least a convergence evaluator must be available"
        self._operator = op
        self._grid = grid
        self._rhs = rhs
        self._dimension = dimension
        self._coarsening_factor = coarsening_factor
        self._convergence_evaluator = convergence_evaluator
        self._performance_evaluator = performance_evaluator
        self._epsilon = epsilon
        self._infinity = infinity
        self._diagonal = base.Diagonal(self._operator)
        self._symbols = set()
        self._types = set()
        self._symbol_types = {}
        self._symbol_names = {}
        self._primitive_set = gp.PrimitiveSetTyped("main", [], types.generate_matrix_type(self._grid.shape))
        self._init_terminals()
        self._init_operators()
        self._init_creator()
        self._init_toolbox()
        self._weight_optimizer = WeightOptimizer(self)

    def _init_terminals(self):
        A = self._operator
        u = self._grid
        f = self._rhs
        D = self._diagonal

        identity_matrix = base.Identity(A.shape, self.dimension)
        # Add primitives to set
        self.add_terminal(A, types.generate_matrix_type(A.shape), 'A')
        self.add_terminal(u, types.generate_matrix_type(u.shape), 'u')

        self.add_terminal(identity_matrix, types.generate_diagonal_matrix_type(A.shape), 'I')
        self.add_terminal(D, types.generate_diagonal_matrix_type(A.shape), 'A_d')
        self.add_terminal(base.LowerTriangle(A), types.generate_matrix_type(A.shape), 'A_l')
        self.add_terminal(base.UpperTriangle(A), types.generate_matrix_type(A.shape), 'A_u')

        #TODO quick hack for testing here
        interpolation_stencil_entries = [
            ((-1, -1), 1.0/4),
            (( 0, -1), 1.0/2),
            (( 1, -1), 1.0/4),
            ((-1,  0), 1.0/2),
            (( 0,  0), 1.0),
            (( 1,  0), 1.0/2),
            ((-1,  1), 1.0/4),
            (( 0,  1), 1.0/2),
            (( 1,  1), 1.0/4),
        ]

        restriction_stencil_entries = [
            ((-1, -1), 1.0/16),
            (( 0, -1), 1.0/8),
            (( 1, -1), 1.0/16),
            ((-1,  0), 1.0/8),
            (( 0,  0), 1.0/4),
            (( 1,  0), 1.0/8),
            ((-1,  1), 1.0/16),
            (( 0,  1), 1.0/8),
            (( 1,  1), 1.0/16),
        ]
        from functools import reduce
        import operator
        accumulated_coarsening_factor = reduce(operator.mul, self.coarsening_factor)
        coarse_grid = multigrid.get_coarse_grid(u, accumulated_coarsening_factor)
        coarse_operator = multigrid.get_coarse_operator(A, accumulated_coarsening_factor)
        interpolation = multigrid.get_interpolation(u, coarse_grid, Stencil(interpolation_stencil_entries))
        restriction = multigrid.get_restriction(u, coarse_grid, Stencil(restriction_stencil_entries))

        self.add_terminal(base.Zero(A), types.generate_matrix_type(coarse_grid.shape))
        self.add_terminal(multigrid.CoarseGridSolver(coarse_grid), types.generate_matrix_type(coarse_operator.shape), 'S_coarse')
        self.add_terminal(interpolation, types.generate_matrix_type(interpolation.shape), 'P')
        self.add_terminal(restriction, types.generate_matrix_type(restriction.shape), 'R')

        self._coarsening_factor = accumulated_coarsening_factor
        self._coarse_grid = coarse_grid
        self._coarse_operator = coarse_operator
        self._interpolation = interpolation
        self._restriction = restriction

    def _init_operators(self):
        A = self._operator
        u = self._grid
        OperatorType = types.generate_matrix_type(A.shape)
        GridType = types.generate_matrix_type(u.shape)
        DiagonalOperatorType = types.generate_diagonal_matrix_type(self._diagonal.shape)

        # Add primitives to full set
        self.add_operator(base.add, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, 'add')
        self.add_operator(base.add, [OperatorType, OperatorType], OperatorType, 'add')

        self.add_operator(base.sub, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, 'sub')
        self.add_operator(base.sub, [OperatorType, OperatorType], OperatorType, 'sub')

        self.add_operator(base.mul, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, 'mul')
        self.add_operator(base.mul, [OperatorType, OperatorType], OperatorType, 'mul')

        self.add_operator(base.inv, [DiagonalOperatorType], DiagonalOperatorType, 'inverse')

        # Correction

        correct = functools.partial(multigrid.correct, self._operator, self._rhs)
        self.add_operator(correct, [OperatorType, GridType], GridType, 'correct')

        # Multigrid recipes
        InterpolationType = types.generate_matrix_type(self._interpolation.shape)
        RestrictionType = types.generate_matrix_type(self._restriction.shape)
        CoarseOperatorType = types.generate_matrix_type(self._coarse_operator.shape)

        # Create intergrid operators
        self.add_operator(base.mul, [CoarseOperatorType, RestrictionType], RestrictionType, 'mul')
        self.add_operator(base.mul, [InterpolationType, CoarseOperatorType], InterpolationType, 'mul')
        self.add_operator(base.mul, [InterpolationType, RestrictionType], OperatorType, 'mul')

        # Dummy operations
        def noop(A):
            return A

        self.add_operator(noop, [CoarseOperatorType], CoarseOperatorType, 'noop')
        self.add_operator(noop, [RestrictionType], RestrictionType, 'noop')
        self.add_operator(noop, [InterpolationType], InterpolationType, 'noop')

    @staticmethod
    def _init_creator():
        creator.create("Fitness", deap.base.Fitness, weights=(-1.0,))
        creator.create("Individual", AST, fitness=creator.Fitness)

    def _init_toolbox(self):
        self._toolbox = deap.base.Toolbox()
        self._toolbox.register("expression", gp.genHalfAndHalf, pset=self._primitive_set, min_=1, max_=5)
        self._toolbox.register("individual", tools.initIterate, creator.Individual, self._toolbox.expression)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("evaluate", self.evaluate)
        self._toolbox.register("select", tools.selTournament, tournsize=4)
        self._toolbox.register("mate", gp.cxOnePoint)
        self._toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
        self._toolbox.register("mutate", gp.mutUniform, expr=self._toolbox.expr_mut, pset=self._primitive_set)

        self._toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=15))
        self._toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=15))

    def set_matrix_type(self, symbol, matrix_type):
        self._symbol_types[symbol] = matrix_type

    def get_symbol_type(self, symbol):
        return self._symbol_types[symbol]

    @property
    def operator(self) -> base.Operator:
        return self._operator

    @property
    def grid(self) -> base.Grid:
        return self._grid

    @property
    def rhs(self) -> base.Grid:
        return self._rhs

    @property
    def dimension(self):
        return self._dimension

    @property
    def coarsening_factor(self):
        return self._coarsening_factor

    @property
    def convergence_evaluator(self):
        return self._convergence_evaluator

    @property
    def performance_evaluator(self):
        return self._performance_evaluator

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def infinity(self):
        return self._infinity

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
            self._symbol_names[symbol] = symbol.name
            self._primitive_set.addTerminal(symbol, matrix_type)

    def add_operator(self, primitive, argument_types, result_type, name: str):
        for argument_type in argument_types:
            self._types.add(argument_type)
        self._types.add(result_type)
        self._primitive_set.addPrimitive(primitive, argument_types, result_type, name)

    def generate_individual(self):
        return self._toolbox.individual()

    def compile_expression(self, expression):
        return gp.compile(expression, self._primitive_set)

    @staticmethod
    def get_iteration_matrix(expression, grid, rhs):
        from evostencils.expressions.transformations import propagate_zero, substitute_entity
        tmp = substitute_entity(expression, rhs, base.Zero(rhs.shape))
        tmp = propagate_zero(tmp)
        return substitute_entity(tmp, grid, base.Identity(grid.shape))

    def evaluate(self, individual):
        import math
        expression = transformations.fold_intergrid_operations(self.compile_expression(individual))
        if individual.weights is not None:
            expression = transformations.set_weights(expression, individual.weights)
        iteration_matrix = self.get_iteration_matrix(expression, self.grid, self.rhs)
        spectral_radius = self.convergence_evaluator.compute_spectral_radius(iteration_matrix)
        if spectral_radius == 0.0:
            return self.infinity,
        else:
            if self._performance_evaluator is not None:
                runtime = self.performance_evaluator.estimate_runtime(expression)
                if spectral_radius < 1.0:
                    return math.log(self.epsilon) / math.log(spectral_radius) * runtime,
                else:
                    return self.infinity * spectral_radius * runtime,
            else:
                return spectral_radius,

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
        for individual in hof:
            weights, spectral_radius = self.optimize_weights(individual)
            individual.set_weights(weights)
            individual.fitness = creator.Fitness(values=self.evaluate(individual))
        return pop, log, hof

    def default_optimization(self, population, generations, crossover_probability, mutation_probability):
        return self.harm_gp(population, generations, crossover_probability, mutation_probability)

    def optimize_weights(self, individual):
        expression = transformations.fold_intergrid_operations(self.compile_expression(individual))
        weights = transformations.obtain_weights(expression)
        best_individual = self._weight_optimizer.optimize(expression, len(weights), 50)
        best_weights = list(best_individual)
        spectral_radius = best_individual.fitness.values[0]
        return best_weights, spectral_radius

    @staticmethod
    def visualize_tree(individual, filename):
        import pygraphviz as pgv
        nodes, edges, labels = gp.graph(individual)
        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")
        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]
        g.draw(f"{filename}.png", "png")





