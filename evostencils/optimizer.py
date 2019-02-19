import numpy as np
import deap.base
from deap import gp, creator, tools, algorithms
import random
import pickle
import os.path
from evostencils.initialization import multigrid
import evostencils.expressions.base as base
import evostencils.expressions.transformations as transformations
from evostencils.deap_extension import genGrow, AST, PrimitiveSetTyped
from evostencils.weight_optimizer import WeightOptimizer
from evostencils.types import level_control


class CheckPoint:
    def __init__(self, min_level, max_level, generation, program, solver, population):
        self.min_level = min_level
        self.max_level = max_level
        self.generation = generation
        self.program = program
        self.solver = solver
        self.population = population

    def dump_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)


def load_checkpoint_from_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


class Optimizer:
    def __init__(self, op: base.Operator, grid: base.Grid, rhs: base.Grid, dimension, coarsening_factor,
                 interpolation, restriction, min_level, max_level, convergence_evaluator=None, performance_evaluator=None,
                 program_generator=None, epsilon=1e-20, infinity=1e100, checkpoint_directory_path='./'):
        assert convergence_evaluator is not None, "At least a convergence evaluator must be available"
        self._operator = op
        self._grid = grid
        self._rhs = rhs
        self._dimension = dimension
        self._coarsening_factor = coarsening_factor
        self._interpolation = interpolation
        self._restriction = restriction
        self._max_level = max_level
        self._min_level = min_level
        self._convergence_evaluator = convergence_evaluator
        self._performance_evaluator = performance_evaluator
        self._program_generator = program_generator
        self._epsilon = epsilon
        self._infinity = infinity
        self._checkpoint_directory_path = checkpoint_directory_path
        self._FinishedType = level_control.generate_finished_type()
        self._NotFinishedType = level_control.generate_not_finished_type()
        self._init_creator()
        self._weight_optimizer = WeightOptimizer(self)

    @staticmethod
    def _init_creator():
        creator.create("Fitness", deap.base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", AST, fitness=creator.Fitness)

    def _init_toolbox(self, pset):
        self._toolbox = deap.base.Toolbox()
        self._toolbox.register("expression", genGrow, pset=pset, min_height=5, max_height=10)
        self._toolbox.register("individual", tools.initIterate, creator.Individual, self._toolbox.expression)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("evaluate", self.evaluate, pset=pset)
        self._toolbox.register("select", tools.selNSGA2, nd='log')
        # self._toolbox.register("mate", gp.cxOnePoint)
        self._toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.2)
        self._toolbox.register("expr_mut", genGrow, pset=pset, min_height=1, max_height=8)
        self._toolbox.register("mutate", gp.mutUniform, expr=self._toolbox.expr_mut, pset=pset)
        # self._toolbox.register("mutInsert", gp.mutInsert, pset=pset)

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
    def min_level(self):
        return self._min_level

    @property
    def max_level(self):
        return self._max_level

    @property
    def convergence_evaluator(self):
        return self._convergence_evaluator

    @property
    def performance_evaluator(self):
        return self._performance_evaluator

    @property
    def program_generator(self):
        return self._program_generator

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def infinity(self):
        return self._infinity

    def generate_individual(self):
        return self._toolbox.individual()

    def compile_individual(self, individual, pset):
        return gp.compile(individual, pset)

    def evaluate(self, individual, pset):
        import numpy, math
        if len(individual) > 150:
            return self.infinity, self.infinity
        try:
            expression1, expression2 = self.compile_individual(individual, pset)
        except MemoryError:
            return self.infinity, self.infinity

        expression = expression1
        iteration_matrix = transformations.get_iteration_matrix(expression)
        spectral_radius = self.convergence_evaluator.compute_spectral_radius(iteration_matrix)

        if spectral_radius == 0.0 or math.isnan(spectral_radius) \
                or math.isinf(spectral_radius) or numpy.isinf(spectral_radius) or numpy.isnan(spectral_radius):
            return self.infinity, self.infinity
        else:
            if self._performance_evaluator is not None:
                try:
                    runtime = self.performance_evaluator.estimate_runtime(expression) * 1e3 # ms
                except RuntimeError as _:
                    return self.infinity, self.infinity
                return spectral_radius, runtime
            else:
                return spectral_radius, self.infinity

    def random_search(self, initial_population_size, generations, mu_, lambda_):
        random.seed()
        pop = self._toolbox.population(n=initial_population_size)
        hof = tools.ParetoFront()

        stats_fit1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fit2 = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(convergence=stats_fit1, runtime=stats_fit2, size=stats_size)

        def mean(xs):
            avg = 0
            for x in xs:
                if x < self.infinity:
                    avg += x
            avg = avg / len(xs)
            return avg

        mstats.register("avg", mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        toolbox = self._toolbox
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))

        record = mstats.compile(pop) if mstats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        # Begin the generational process
        for gen in range(1, generations + 1):
            # Vary the population
            offspring = toolbox.population(n=lambda_)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = offspring
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            hof.update(pop)

            # Select the next generation population
            pop = toolbox.select(pop + offspring, mu_)
            record = mstats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

        return pop, logbook, hof

    def nsgaII(self, initial_population_size, generations, mu_, lambda_, crossover_probability, mutation_probability,
               min_level, max_level, program, solver, checkpoint_frequency=5, initial_population=None):
        random.seed()
        if initial_population is None:
            pop = self._toolbox.population(n=initial_population_size)
        else:
            pop = initial_population
        hof = tools.ParetoFront()

        stats_fit1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fit2 = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(convergence=stats_fit1, runtime=stats_fit2, size=stats_size)

        def mean(xs):
            avg = 0
            for x in xs:
                if x < self.infinity:
                    avg += x
            avg = avg / len(xs)
            return avg

        mstats.register("avg", mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        toolbox = self._toolbox
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))

        record = mstats.compile(pop) if mstats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        # Begin the generational process
        for gen in range(1, generations + 1):
            # Vary the population
            offspring = tools.selTournamentDCD(pop, 4 * (lambda_ // 4))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                rnd = random.random()
                if rnd <= crossover_probability:
                    toolbox.mate(ind1, ind2)
                elif rnd <= crossover_probability + mutation_probability:
                    toolbox.mutate(ind1)
                    toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            hof.update(pop)
            if gen % checkpoint_frequency == 0:
                if solver is not None:
                    transformations.invalidate_expression(solver.iteration_matrix)
                    transformations.invalidate_expression(solver)
                checkpoint = CheckPoint(min_level, max_level, gen, program, solver, pop)
                checkpoint.dump_to_file(f'{self._checkpoint_directory_path}/checkpoint.p')
            # Select the next generation population
            pop = toolbox.select(pop + offspring, mu_)
            record = mstats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

        return pop, logbook, hof

    def default_optimization(self, gp_mu=100, gp_lambda=100, gp_generations=20, gp_crossover_probability=0.7,
                             gp_mutation_probability=0.3, es_lambda=10, es_generations=50, required_convergence=0.5,
                             restart_from_checkpoint=False):
        from evostencils.expressions import multigrid as mg_exp
        import math
        gp_generations_remaining = gp_generations
        levels = self.max_level - self.min_level
        levels_per_run = 2
        grids = [self.grid]
        right_hand_sides = [self.rhs]
        for i in range(1, levels + 1):
            grids.append(mg_exp.get_coarse_grid(grids[-1], self.coarsening_factor))
            right_hand_sides.append(mg_exp.get_coarse_rhs(right_hand_sides[-1], self.coarsening_factor))
        cgs_expression = None
        storages = self._program_generator.generate_storage(levels)
        checkpoint = None
        checkpoint_file_path = f'{self._checkpoint_directory_path}/checkpoint.p'
        solver_program = ""
        if restart_from_checkpoint and os.path.isfile(checkpoint_file_path):
            try:
                checkpoint = load_checkpoint_from_file(checkpoint_file_path)
                solver_program = checkpoint.program
            except pickle.PickleError as _:
                restart_from_checkpoint = False
        else:
            restart_from_checkpoint = False
        pops = []
        stats = []
        boilerplate_program = self._program_generator.generate_boilerplate(storages, self.dimension, self.epsilon)
        for i in range(levels - levels_per_run, -1, -levels_per_run):
            min_level = i
            max_level = i + levels_per_run - 1
            evaluation_boilerplate = self._program_generator.generate_boilerplate(storages, self.dimension, self.epsilon, min_level)
            initial_population = None
            if restart_from_checkpoint:
                if min_level == checkpoint.min_level and max_level == checkpoint.max_level:
                    initial_population = checkpoint.population
                    cgs_expression = checkpoint.solver
                    gp_generations_remaining = gp_generations - checkpoint.generation
                elif min_level > checkpoint.min_level:
                    continue
                else:
                    initial_population = None
                    gp_generations_remaining = gp_generations
            grid = grids[i]
            rhs = right_hand_sides[i]
            operator = mg_exp.get_coarse_operator(self.operator, grid)
            interpolation = mg_exp.get_interpolation(grids[i], grids[i+levels_per_run-1], self.interpolation.stencil_generator)
            restriction = mg_exp.get_restriction(grids[i], grids[i+levels_per_run-1], self.restriction.stencil_generator)
            pset = multigrid.generate_primitive_set(operator, grid, rhs, self.dimension, self.coarsening_factor,
                                                    interpolation, restriction,
                                                    coarse_grid_solver_expression=cgs_expression,
                                                    depth=levels_per_run, LevelFinishedType=self._FinishedType,
                                                    LevelNotFinishedType=self._NotFinishedType)
            self._init_toolbox(pset)
            pop, log, hof = self.nsgaII(10 * gp_mu, gp_generations_remaining, gp_mu, gp_lambda, gp_crossover_probability,
                                        gp_mutation_probability, min_level, max_level, solver_program, cgs_expression,
                                        checkpoint_frequency=5, initial_population=initial_population)
            # pop, log, hof = self.random_search(population_size * 10, generations, mu_, lambda_)
            pops.append(pop)
            stats.append(log)

            def key_function(ind):
                rho = ind.fitness.values[0]
                if rho <= required_convergence:
                    tmp = math.log(self.epsilon) / math.log(rho)
                    tmp = tmp * ind.fitness.values[1]
                    return tmp
                else:
                    return rho * self.infinity
            hof = sorted(hof, key=key_function)
            best_time = self.infinity
            best_convergence_factor = self.infinity
            best_individual = hof[0]
            base_program = evaluation_boilerplate + solver_program
            if self._program_generator.compiler_available:
                count = 0
                for j in range(len(hof)):
                    spectral_radius = hof[j].fitness.values[0]
                    if spectral_radius > required_convergence or count == 50:
                        break
                    if j < len(hof) - 1 and abs(hof[j].fitness.values[0] - hof[j+1].fitness.values[0]) < self.epsilon and \
                            abs(hof[j].fitness.values[1] - hof[j+1].fitness.values[1] < self.epsilon):
                        continue
                    ind = hof[j]
                    expression = self.compile_individual(ind, pset)[0]
                    evaluation_program = base_program + self._program_generator.generate_cycle_function(expression, storages)
                    # print(evaluation_program)
                    self._program_generator.write_program_to_file(evaluation_program)
                    time_to_solution, convergence_factor = self._program_generator.execute()
                    print(f'Time to solution: {time_to_solution}')
                    self._program_generator.invalidate_storages(storages)
                    if time_to_solution < best_time and convergence_factor < 1:
                        best_time = time_to_solution
                        best_convergence_factor = convergence_factor
                        best_individual = ind
                    count += 1
            print(f"Best individual: ({best_convergence_factor}), ({best_individual.fitness.values[1]})")
            best_expression = self.compile_individual(best_individual, pset)[0]
            cgs_expression = best_expression
            cgs_expression.evaluate = False
            optimized_weights, optimized_convergence_factor = self.optimize_weights(cgs_expression, es_lambda,
                                                                                    es_generations, base_program, storages)
            if optimized_convergence_factor < best_convergence_factor:
                self._weight_optimizer.restrict_weights(optimized_weights, 0.0, 2.0)
                transformations.set_weights(cgs_expression, optimized_weights)
                print(f"Best individual: ({optimized_convergence_factor}), ({best_individual.fitness.values[1]})")
            iteration_matrix = transformations.get_iteration_matrix(cgs_expression)
            # print(repr(iteration_matrix))
            self.convergence_evaluator.compute_spectral_radius(iteration_matrix)
            self.performance_evaluator.estimate_runtime(cgs_expression)
            try:
                solver_program += self._program_generator.generate_cycle_function(cgs_expression, storages)
            except Exception as e:
                print('Ungeneratable program')
                print(e)

        return boilerplate_program + solver_program, pops, stats

    def optimize_weights(self, expression, lambda_, generations, base_program=None, storages=None):
        # expression = self.compile_expression(individual)
        weights = transformations.obtain_weights(expression)
        best_individual = self._weight_optimizer.optimize(expression, len(weights), lambda_, generations, base_program, storages)
        best_weights = list(best_individual)
        spectral_radius, = best_individual.fitness.values
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
    def plot_multiobjective_data(generations, convergence_data, runtime_data, label1, label2):
        from matplotlib.ticker import MaxNLocator
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(generations, convergence_data, "b-", label=f"{label1}")
        ax1.set_xlabel("Generation")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_ylabel("Spectral Radius", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        ax2 = ax1.twinx()
        line2 = ax2.plot(generations, runtime_data, "r-", label=f"{label2}")
        ax2.set_ylabel("Runtime", color="r")
        for tl in ax2.get_yticklabels():
            tl.set_color("r")

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="top right")

        plt.show()

    @staticmethod
    def plot_minimum_fitness(logbook):
        gen = logbook.select("gen")
        convergence_mins = logbook.chapters["convergence"].select("min")
        runtime_mins = logbook.chapters["runtime"].select("min")
        Optimizer.plot_multiobjective_data(gen, convergence_mins, runtime_mins, 'Minimum Spectral Radius', 'Minimum Runtime')

    @staticmethod
    def plot_average_fitness(logbook):
        gen = logbook.select("gen")
        convergence_avgs = logbook.chapters["convergence"].select("avg")
        runtime_avgs = logbook.chapters["runtime"].select("avg")
        Optimizer.plot_multiobjective_data(gen, convergence_avgs, runtime_avgs, 'Average Spectral Radius', 'Average Runtime')

    @staticmethod
    def plot_pareto_front(pop):
        import matplotlib.pyplot as plt
        import numpy
        pop.sort(key=lambda x: x.fitness.values)

        front = numpy.array([ind.fitness.values for ind in pop])
        plt.scatter(front[:, 0], front[:, 1], c="b")
        plt.xlabel("Spectral Radius")
        plt.ylabel("Runtime")
        plt.axis("tight")
        plt.show()

    @staticmethod
    def dump_population(pop, file_name):
        import pickle
        with open(file_name, 'wb') as file:
            pickle.dump(pop, file)



