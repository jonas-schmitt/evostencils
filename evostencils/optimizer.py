import numpy as np
import deap.base
from deap import gp, creator, tools, algorithms
import random
from evostencils.initialization import multigrid
import evostencils.expressions.base as base
import evostencils.expressions.transformations as transformations
from evostencils.deap_extension import generate_tree_with_minimum_height
from evostencils.weight_optimizer import WeightOptimizer
from evostencils.exastencils.generation import ProgramGenerator
from evostencils.types import multiple


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
                 interpolation, restriction, levels, convergence_evaluator=None, performance_evaluator=None,
                 epsilon=1e-10, infinity=1e100):
        assert convergence_evaluator is not None, "At least a convergence evaluator must be available"
        self._operator = op
        self._grid = grid
        self._rhs = rhs
        self._dimension = dimension
        self._coarsening_factor = coarsening_factor
        self._interpolation = interpolation
        self._restriction = restriction
        self._levels = levels
        self._convergence_evaluator = convergence_evaluator
        self._performance_evaluator = performance_evaluator
        self._epsilon = epsilon
        self._infinity = infinity
        self._LevelFinishedType = multiple.generate_new_type('LevelFinishedType')
        self._LevelNotFinishedType = multiple.generate_new_type('LevelNotFinishedType')
        self._init_creator()
        self._weight_optimizer = WeightOptimizer(self)
        self._program_generator = ProgramGenerator('2D_FD_Poisson', '/local/ja42rica/ScalaExaStencil', self.operator, self.grid, self.rhs, self.dimension, self.coarsening_factor, self.interpolation, self.restriction)

    @staticmethod
    def _init_creator():
        creator.create("Fitness", deap.base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", AST, fitness=creator.Fitness)

    def _init_toolbox(self, pset):
        self._toolbox = deap.base.Toolbox()
        self._toolbox.register("expression", generate_tree_with_minimum_height, pset=pset, min_height=5, max_height=10, LevelFinishedType=self._LevelFinishedType, LevelNotFinishedType=self._LevelNotFinishedType)
        self._toolbox.register("individual", tools.initIterate, creator.Individual, self._toolbox.expression)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("evaluate", self.evaluate, pset=pset)
        self._toolbox.register("select", tools.selNSGA2)
        self._toolbox.register("mate", gp.cxOnePoint)
        self._toolbox.register("expr_mut", generate_tree_with_minimum_height, pset=pset, min_height=2, max_height=8, LevelFinishedType=self._LevelFinishedType, LevelNotFinishedType=self._LevelNotFinishedType)
        self._toolbox.register("mutate", gp.mutUniform, expr=self._toolbox.expr_mut, pset=pset)

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
    def interpolation(self):
        return self._interpolation

    @property
    def restriction(self):
        return self._restriction

    @property
    def levels(self):
        return self._levels

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

    def generate_individual(self):
        return self._toolbox.individual()

    def compile_expression(self, expression, pset):
        return gp.compile(expression, pset)

    def evaluate(self, individual, pset):
        import numpy, math
        if len(individual) > 150:
            return self.infinity, self.infinity
        try:
            expression1, expression2 = self.compile_expression(individual, pset)
        except MemoryError:
            return self.infinity, self.infinity

        expression = expression1
        iteration_matrix = transformations.get_iteration_matrix(expression)
        spectral_radius = self.convergence_evaluator.compute_spectral_radius(iteration_matrix)
        # simplified_iteration_matrix = transformations.simplify_iteration_matrix(iteration_matrix)
        # transformations.simplify_iteration_matrix_on_all_levels(simplified_iteration_matrix)
        # simplified_spectral_radius = self.convergence_evaluator.compute_spectral_radius(simplified_iteration_matrix)
        # spectral_radius = simplified_spectral_radius

        if spectral_radius == 0.0 or math.isnan(spectral_radius) \
                or math.isinf(spectral_radius) or numpy.isinf(spectral_radius) or numpy.isnan(spectral_radius):
            return self.infinity, self.infinity
        else:
            if self._performance_evaluator is not None:
                runtime = self.performance_evaluator.estimate_runtime(expression)
                return spectral_radius, runtime
            else:
                return spectral_radius, self.infinity

    def ea_simple(self, population_size, generations, crossover_probability, mutation_probability):
        random.seed()
        pop = self._toolbox.population(n=population_size)
        hof = tools.HallOfFame(10)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, self._toolbox, crossover_probability, mutation_probability, generations,
                                       stats=mstats, halloffame=hof, verbose=True)
        return pop, log, hof

    def gp_harm(self, population_size, generations, crossover_probability, mutation_probability):
        random.seed()
        pop = self._toolbox.population(n=population_size)
        hof = tools.HallOfFame(10)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = gp.harm(pop, self._toolbox, crossover_probability, mutation_probability, generations,
                           alpha=0.05, beta=10, gamma=0.20, rho=0.9, stats=mstats, halloffame=hof, verbose=True)
        return pop, log, hof

    def ea_mu_plus_lambda(self, initial_population_size, generations, mu_, lambda_, crossover_probability, mutation_probability):
        random.seed()
        pop = self._toolbox.population(n=initial_population_size)
        hof = tools.HallOfFame(10)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = algorithms.eaMuPlusLambda(pop, self._toolbox, mu_, lambda_, crossover_probability,
                                             mutation_probability, generations,
                                             stats=mstats, halloffame=hof, verbose=True)
        return pop, log, hof

    def default_optimization(self, population_size, generations, crossover_probability, mutation_probability):
        from evostencils.expressions import multigrid as mg_exp
        levels_per_run = 2
        grids = [self.grid]
        right_hand_sides = [self.rhs]
        for i in range(1, self.levels+1):
            grids.append(mg_exp.get_coarse_grid(grids[-1], self.coarsening_factor))
            right_hand_sides.append(mg_exp.get_coarse_rhs(right_hand_sides[-1], self.coarsening_factor))
        cgs_expression = None
        storages = self._program_generator.generate_storage(self.levels)
        program = self._program_generator.generate_boilerplate(storages, self.levels)
        for i in range(self.levels - levels_per_run, -1, -levels_per_run):
            grid = grids[i]
            rhs = right_hand_sides[i]
            operator = mg_exp.get_coarse_operator(self.operator, grid)
            interpolation = mg_exp.get_interpolation(grids[i], grids[i+1], self.interpolation.stencil_generator)
            restriction = mg_exp.get_restriction(grids[i], grids[i+1], self.restriction.stencil_generator)
            pset = multigrid.generate_primitive_set(operator, grid, rhs, self.dimension, self.coarsening_factor,
                                                    interpolation, restriction,
                                                    coarse_grid_solver_expression=cgs_expression,
                                                    depth=levels_per_run, LevelFinishedType=self._LevelFinishedType,
                                                    LevelNotFinishedType=self._LevelNotFinishedType)
            self._init_toolbox(pset)
            mu_ = population_size
            lambda_ = population_size
            pop, log, hof = self.ea_mu_plus_lambda(population_size * 2, generations, mu_, lambda_, crossover_probability, mutation_probability)
            best_individual = hof[0]
            best_expression = self.compile_expression(best_individual, pset)[0]
            cgs_expression = best_expression
            cgs_expression.evaluate = False
            iteration_matrix = transformations.get_iteration_matrix(cgs_expression)
            # Potentially speeds up the convergence evaluation but leads to slightly different spectral radii
            # iteration_matrix = transformations.simplify_iteration_matrix(iteration_matrix)
            # transformations.simplify_iteration_matrix_on_all_levels(iteration_matrix)
            self.convergence_evaluator.compute_spectral_radius(iteration_matrix)
            self.performance_evaluator.estimate_runtime(cgs_expression)
            program += self._program_generator.generate_cycle_function(best_expression, storages)
        return program

    def optimize_weights(self, expression, iterations=50):
        # expression = self.compile_expression(individual)
        weights = transformations.obtain_weights(expression)
        best_individual = self._weight_optimizer.optimize(expression, len(weights), iterations)
        best_weights = list(best_individual)
        spectral_radius = best_individual.fitness.values
        # print(best_weights)
        # print(spectral_radius)
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

    @staticmethod
    def plot_minimum_fitness(logbook):
        from matplotlib.ticker import MaxNLocator
        gen = logbook.select("gen")
        fit_mins = logbook.chapters["fitness"].select("min")
        size_avgs = logbook.chapters["size"].select("avg")

        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
        ax1.set_xlabel("Generation")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_ylabel("Fitness", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        ax2 = ax1.twinx()
        line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
        ax2.set_ylabel("Size", color="r")
        for tl in ax2.get_yticklabels():
            tl.set_color("r")

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="top right")

        plt.show()





