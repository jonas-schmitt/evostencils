import numpy as np
import deap.base
from deap import gp, creator, tools
import random
import pickle
import os.path
from evostencils.initialization import multigrid as multigrid_initialization
from evostencils.expressions import base, transformations, system
from evostencils.genetic_programming import genGrow, mutNodeReplacement, mutInsert, select_unique_best
import evostencils.optimization.relaxation_factors as relaxation_factor_optimization
from evostencils.types import level_control
import math, numpy
from functools import reduce


class suppress_output(object):
    def __init__(self):
        self.null_file_descriptor = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_file_descriptor = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_file_descriptor[0], 1)
        os.dup2(self.null_file_descriptor[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_file_descriptor[0], 1)
        os.dup2(self.save_file_descriptor[1], 2)
        for fd in self.null_file_descriptor + self.save_file_descriptor:
            os.close(fd)


class CheckPoint:
    def __init__(self, min_level, max_level, generation, program, solver, population, logbooks):
        self.min_level = min_level
        self.max_level = max_level
        self.generation = generation
        self.program = program
        self.solver = solver
        self.population = population
        self.logbooks = logbooks

    def dump_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)


def load_checkpoint_from_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


class Optimizer:
    def __init__(self, dimension, finest_grid, coarsening_factor, min_level, max_level, equations, operators, fields,
                 convergence_evaluator, performance_evaluator=None,
                 program_generator=None, epsilon=1e-10, infinity=1e300, checkpoint_directory_path='./'):
        assert convergence_evaluator is not None, "At least a convergence evaluator must be available"
        self._dimension = dimension
        self._finest_grid = finest_grid
        solution_entries = [base.Approximation(f.name, g) for f, g in zip(fields, finest_grid)]
        self._approximation = system.Approximation('x', solution_entries)
        rhs_entries = [base.RightHandSide(eq.rhs_name, g) for eq, g in zip(equations, finest_grid)]
        self._rhs = system.RightHandSide('b', rhs_entries)
        self._coarsening_factor = coarsening_factor
        self._max_level = max_level
        self._min_level = min_level
        self._equations = equations
        self._operators = operators
        self._fields = fields
        self._convergence_evaluator = convergence_evaluator
        self._performance_evaluator = performance_evaluator
        self._program_generator = program_generator
        self._epsilon = epsilon
        self._infinity = infinity
        self._checkpoint_directory_path = checkpoint_directory_path
        self._FinishedType = level_control.generate_finished_type()
        self._NotFinishedType = level_control.generate_not_finished_type()
        self._init_creator()
        self._weight_optimizer = relaxation_factor_optimization.Optimizer(self)

    @staticmethod
    def _init_creator():
        creator.create("MultiObjectiveFitness", deap.base.Fitness, weights=(-1.0, -1.0))
        creator.create("MultiObjectiveIndividual", gp.PrimitiveTree, fitness=creator.MultiObjectiveFitness)
        creator.create("SingleObjectiveFitness", deap.base.Fitness, weights=(-1.0,))
        creator.create("SingleObjectiveIndividual", gp.PrimitiveTree, fitness=creator.SingleObjectiveFitness)

    def _init_toolbox(self, pset):
        self._toolbox = deap.base.Toolbox()
        self._toolbox.register("expression", genGrow, pset=pset, min_height=10, max_height=20)
        self._toolbox.register("mate", gp.cxOnePoint)

        def mutate(individual, pset):
            operator_choice = random.random()
            if operator_choice < 0.5:
                return mutInsert(individual, 1, 10, pset)
            else:
                return mutNodeReplacement(individual, pset)

        self._toolbox.register("mutate", mutate, pset=pset)

    def _init_multi_objective_toolbox(self, pset):
        self._toolbox.register("individual", tools.initIterate, creator.MultiObjectiveIndividual,
                               self._toolbox.expression)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("evaluate", self.evaluate_multiple_objectives, pset=pset)

    def _init_single_objective_toolbox(self, pset):
        self._toolbox.register("individual", tools.initIterate, creator.SingleObjectiveIndividual,
                               self._toolbox.expression)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("evaluate", self.evaluate_single_objective, pset=pset)

    @property
    def dimension(self):
        return self._dimension

    @property
    def finest_grid(self):
        return self._finest_grid

    @property
    def coarsening_factors(self):
        return self._coarsening_factor

    @property
    def min_level(self):
        return self._min_level

    @property
    def max_level(self):
        return self._max_level

    @property
    def equations(self):
        return self._equations

    @property
    def operators(self):
        return self._operators

    @property
    def fields(self):
        return self._fields

    @property
    def approximation(self):
        return self._approximation

    @property
    def rhs(self):
        return self._rhs

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

    def evaluate_multiple_objectives(self, individual, pset):
        if len(individual) > 150:
            return self.infinity, self.infinity
        with suppress_output():
            try:
                expression1, expression2 = self.compile_individual(individual, pset)
            except MemoryError:
                return self.infinity, self.infinity

        expression = expression1
        with suppress_output():
            spectral_radius = self.convergence_evaluator.compute_spectral_radius(expression)

        if spectral_radius == 0.0 or math.isnan(spectral_radius) \
                or math.isinf(spectral_radius) or numpy.isinf(spectral_radius) or numpy.isnan(spectral_radius):
            return self.infinity, self.infinity
        else:
            if self._performance_evaluator is not None:
                grid_points = sum([reduce(lambda x, y: x * y, g.size) for g in expression.grid])
                runtime = self.performance_evaluator.estimate_runtime(expression) / grid_points * 1e6
                return spectral_radius, runtime
            else:
                return spectral_radius, self.infinity

    def evaluate_single_objective(self, individual, pset):
        if len(individual) > 150:
            return self.infinity,
        with suppress_output():
            try:
                expression1, expression2 = self.compile_individual(individual, pset)
            except MemoryError:
                return self.infinity,

        expression = expression1
        with suppress_output():
            spectral_radius = self.convergence_evaluator.compute_spectral_radius(expression)

        if spectral_radius == 0.0 or math.isnan(spectral_radius) \
                or math.isinf(spectral_radius) or numpy.isinf(spectral_radius) or numpy.isnan(spectral_radius):
            return self.infinity,
        else:
            if self._performance_evaluator is not None:
                if spectral_radius < 0.1:
                    grid_points = sum([reduce(lambda x, y: x * y, g.size) for g in expression.grid])
                    runtime = self.performance_evaluator.estimate_runtime(expression) / grid_points * 1e6
                    return math.log(self.epsilon) / math.log(spectral_radius) * runtime,
                else:
                    return spectral_radius * math.sqrt(self.infinity),
            else:
                return spectral_radius,

    def ea_mu_plus_lambda(self, initial_population_size, generations, mu_, lambda_,
                          crossover_probability, mutation_probability, min_level, max_level,
                          program, solver, logbooks, checkpoint_frequency, checkpoint, mstats, hof):
        random.seed()
        use_checkpoint = False
        if checkpoint is not None:
            if mu_ == len(checkpoint.population):
                use_checkpoint = True
            else:
                print(f'Could not restart from checkpoint. Checkpoint population size is {len(checkpoint.population)} '
                      f'but the required size is {mu_}.', flush=True)
        if use_checkpoint:
            population = checkpoint.population
            min_generation = checkpoint.generation
        else:
            population = self._toolbox.population(n=initial_population_size)
            min_generation = 0
        max_generation = generations

        if use_checkpoint:
            logbook = logbooks[-1]
        else:
            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
            logbooks.append(logbook)

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        toolbox = self._toolbox
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        population = toolbox.select(population, len(population))
        if hof is not None:
            hof.update(population)
        record = mstats.compile(population) if mstats is not None else {}
        logbook.record(gen=min_generation, nevals=len(invalid_ind), **record)
        print(logbook.stream, flush=True)
        # Begin the generational process
        count = 0
        for gen in range(min_generation + 1, max_generation + 1):
            # Vary the population
            offspring = []
            for i in range(lambda_ // 2):
                ind1, ind2 = map(toolbox.clone, toolbox.select(population, 2))
                operator_choice = random.random()
                if operator_choice < crossover_probability:
                    child1, child2 = toolbox.mate(ind1, ind2)
                elif operator_choice < crossover_probability + mutation_probability + self.epsilon:
                    child1, = toolbox.mutate(ind1)
                    child2, = toolbox.mutate(ind2)
                else:
                    child1 = ind1
                    child2 = ind2
                del child1.fitness.values, child2.fitness.values
                offspring.append(child1)
                offspring.append(child2)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            if hof is not None:
                hof.update(offspring)
            if gen % checkpoint_frequency == 0:
                if solver is not None:
                    transformations.invalidate_expression(solver)
                logbooks[-1] = logbook
                checkpoint = CheckPoint(min_level, max_level, gen, program, solver, population, logbooks)
                try:
                    checkpoint.dump_to_file(f'{self._checkpoint_directory_path}/checkpoint.p')
                except (pickle.PickleError, TypeError) as e:
                    print(e, flush=True)
                    print('Skipping checkpoint', flush=True)
            # Select the next generation population
            population[:] = toolbox.select(population + offspring, mu_)
            record = mstats.compile(population)
            if len(population[0].fitness.values) == 1:
                if record['fitness']['std'] < self.epsilon:
                    count += 1
                else:
                    count = 0
                if count >= 5 and hof is not None:
                    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
                    print(logbook.stream, flush=True)
                    print("Population converged", flush=True)
                    return population, logbook, hof
            # Update the statistics with the new population
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream, flush=True)

        return population, logbook, hof

    def SOGP(self, pset, initial_population_size, generations, mu_, lambda_,
             crossover_probability, mutation_probability, min_level, max_level,
             program, solver, logbooks, checkpoint_frequency=5, checkpoint=None):
        print("Running Single-Objective Genetic Programming", flush=True)
        self._init_single_objective_toolbox(pset)
        self._toolbox.register("select", select_unique_best)
        self._toolbox.register("select_for_mating", tools.selTournament, tournsize=4)
        # self._toolbox.register("select", tools.selBest)
        # self._toolbox.register("select_for_mating", tools.selTournament, tournsize=2)

        def normalize_fitness(ind):
            if math.pow(self.infinity, 1/4) < ind.fitness.values[0] < self.infinity:
                expression, _ = self.compile_individual(ind, pset)
                grid_points = sum([reduce(lambda x, y: x * y, g.size) for g in expression.grid])
                spectral_radius = ind.fitness.values[0] / math.sqrt(self.infinity)
                runtime = self.performance_evaluator.estimate_runtime(expression) / grid_points * 1e6
                return math.log(self.epsilon) / math.log(spectral_radius) * runtime
            else:
                return ind.fitness.values[0]
        stats_fit = tools.Statistics(normalize_fitness)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

        def mean(xs):
            avg = 0
            for x in xs:
                if 0 < x < math.pow(self.infinity, 0.25):
                    avg += x
            avg = avg / len(xs)
            return avg

        def minimum(xs):
            curr = xs[0]
            for x in xs[1:]:
                if 0 < x < math.pow(self.infinity, 0.25) and x < curr:
                    curr = x
            return curr

        def maximum(xs):
            curr = xs[0]
            for x in xs[1:]:
                if 0 < x < math.pow(self.infinity, 0.25) and x > curr:
                    curr = x
            return curr

        def _ss(data):
            c = mean(data)
            ss = sum((x-c)**2 for x in data if 0 < x < math.pow(self.infinity, 0.25))
            return ss

        def stddev(data, ddof=0):
            n = len(data)
            if n < 2:
                raise ValueError('variance requires at least two data points')
            ss = _ss(data)
            pvar = ss/(n-ddof)
            return pvar**0.5

        mstats.register("avg", mean)
        mstats.register("std", stddev)
        mstats.register("min", minimum)
        mstats.register("max", maximum)
        hof = tools.HallOfFame(100, similar=lambda a, b: a.fitness.values[0] == b.fitness.values[0])
        return self.ea_mu_plus_lambda(initial_population_size, generations, mu_, lambda_,
                                      crossover_probability, mutation_probability, min_level, max_level,
                                      program, solver, logbooks, checkpoint_frequency, checkpoint, mstats, hof)

    def NSGAII(self, pset, initial_population_size, generations, mu_, lambda_,
               crossover_probability, mutation_probability, min_level, max_level,
               program, solver, logbooks, checkpoint_frequency=5, checkpoint=None):
        print("Running NSGA-II Genetic Programming", flush=True)
        self._init_multi_objective_toolbox(pset)
        self._toolbox.register("select", tools.selNSGA2, nd='log')
        self._toolbox.register("select_for_mating", tools.selTournamentDCD)

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
        hof = tools.ParetoFront(similar=lambda a, b: a.fitness.values[0] == b.fitness.values[0] and a.fitness.values[1] == b.fitness.values[1])

        return self.ea_mu_plus_lambda(initial_population_size, generations, mu_, lambda_,
                                      crossover_probability, mutation_probability, min_level, max_level,
                                      program, solver, logbooks, checkpoint_frequency, checkpoint, mstats, hof)

    def NSGAIII(self, pset, initial_population_size, generations, mu_, lambda_, crossover_probability, mutation_probability,
                min_level, max_level, program, solver, logbooks, checkpoint_frequency=5, checkpoint=None):
        print("Running NSGA-III Genetic Programming", flush=True)
        self._init_multi_objective_toolbox(pset)
        ref_points = tools.uniform_reference_points(2, 12)
        self._toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
        self._toolbox.register("select_for_mating", tools.selRandom)

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
        hof = tools.ParetoFront(similar=lambda a,b: a.fitness.values[0] == b.fitness.values[0] and a.fitness.values[1] == b.fitness.values[1])

        return self.ea_mu_plus_lambda(initial_population_size, generations, mu_, lambda_,
                                      crossover_probability, mutation_probability, min_level, max_level,
                                      program, solver, logbooks, checkpoint_frequency, checkpoint, mstats, hof)

    def SPEAII(self, pset, initial_population_size, generations, mu_, lambda_, crossover_probability, mutation_probability,
               min_level, max_level, program, solver, logbooks, checkpoint_frequency=5, checkpoint=None):
        print("Running SPEA-II Genetic Programming", flush=True)
        self._init_multi_objective_toolbox(pset)
        self._toolbox.register("select", tools.selSPEA2)
        self._toolbox.register("select_for_mating", tools.selRandom)

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
        hof = tools.ParetoFront(similar=lambda a,b: a.fitness.values[0] == b.fitness.values[0] and a.fitness.values[1] == b.fitness.values[1])

        return self.ea_mu_plus_lambda(initial_population_size, generations, mu_, lambda_,
                                      crossover_probability, mutation_probability, min_level, max_level,
                                      program, solver, logbooks, checkpoint_frequency, checkpoint, mstats, hof)

    def evolutionary_optimization(self, levels_per_run=2, gp_mu=100, gp_lambda=100, gp_generations=50,
                                  gp_crossover_probability=0.5, gp_mutation_probability=0.5, es_generations=100,
                                  required_convergence=0.2,
                                  restart_from_checkpoint=False, maximum_block_size=3, optimization_method=None):

        levels = self.max_level - self.min_level
        approximations = [self.approximation]
        right_hand_sides = [self.rhs]
        for i in range(1, levels + 1):
            approximations.append(system.get_coarse_approximation(approximations[-1], self.coarsening_factors))
            right_hand_sides.append(system.get_coarse_rhs(right_hand_sides[-1], self.coarsening_factors))
        best_expression = None
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
        logbooks = []
        storages = self._program_generator.generate_storage(self.min_level, self.max_level, self.finest_grid)
        for i in range(levels - levels_per_run, -1, -levels_per_run):
            min_level = self.max_level - (i + levels_per_run - 1)
            max_level = self.max_level - i
            pass_checkpoint = False
            if restart_from_checkpoint:
                if min_level == checkpoint.min_level and max_level == checkpoint.max_level:
                    best_expression = checkpoint.solver
                    pass_checkpoint = True
                    logbooks = checkpoint.logbooks
                elif min_level < checkpoint.min_level:
                    continue
            approximation = approximations[i]

            self._convergence_evaluator.reinitialize_lfa_grids(approximation.grid)

            rhs = right_hand_sides[i]
            pset = multigrid_initialization.generate_primitive_set(approximation, rhs, self.dimension, self.coarsening_factors,
                                                                   max_level, self.equations, self.operators, self.fields,
                                                                   maximum_block_size=maximum_block_size,
                                                                   coarse_grid_solver_expression=best_expression,
                                                                   depth=levels_per_run, LevelFinishedType=self._FinishedType,
                                                                   LevelNotFinishedType=self._NotFinishedType)
            self._init_toolbox(pset)
            tmp = None
            if pass_checkpoint:
                tmp = checkpoint
            if optimization_method is None:
                pop, log, hof = self.SOGP(pset, 10 * gp_mu, gp_generations, gp_mu, gp_lambda, gp_crossover_probability,
                                          gp_mutation_probability, min_level, max_level,
                                          solver_program, best_expression, logbooks,
                                          checkpoint_frequency=5, checkpoint=tmp)
            else:
                pop, log, hof = optimization_method(pset, 10 * gp_mu, gp_generations, gp_mu, gp_lambda,
                                                    gp_crossover_probability, gp_mutation_probability,
                                                    min_level, max_level, solver_program, best_expression, logbooks,
                                                    checkpoint_frequency=5, checkpoint=tmp)

            pops.append(pop)

            hof = sorted(hof, key=lambda ind: ind.fitness.values[len(ind.fitness.values)-1])
            best_time = self.infinity
            best_convergence_factor = self.infinity
            self.program_generator._counter = 0
            self.program_generator._average_generation_time = 0
            self.program_generator.initialize_code_generation(max_level)
            try:
                for j in range(0, min(100, len(hof))):
                    individual = hof[j]
                    expression = self.compile_individual(individual, pset)[0]

                    time, convergence_factor = \
                        self._program_generator.generate_and_evaluate(expression, storages, min_level, max_level,
                                                                      solver_program, number_of_samples=100)
                    estimated_convergence, _ = self.evaluate_multiple_objectives(individual, pset)
                    print(f'Time: {time}, Estimated convergence factor: {estimated_convergence}, '
                          f'Measured convergence factor: {convergence_factor}', flush=True)
                    if time < best_time and \
                            ((i == 0 and convergence_factor < 0.9) or convergence_factor < required_convergence):
                        best_expression = expression
                        best_time = time
                        best_convergence_factor = convergence_factor

            except (KeyboardInterrupt, Exception) as e:
                self.program_generator.restore_files()
                raise e
            self.program_generator.restore_files()
            if best_expression is None:
                raise RuntimeError("Optimization failed")
            print(f"Best time: {best_time}, Best convergence factor: {best_convergence_factor}", flush=True)
            relaxation_factors, _ = self.optimize_relaxation_factors(best_expression, es_generations,
                                                                     min_level, max_level, solver_program, storages)
            relaxation_factor_optimization.set_relaxation_factors(best_expression, relaxation_factors)

            cycle_function = self.program_generator.generate_cycle_function(best_expression, storages, min_level,
                                                                            max_level, self.max_level)
            solver_program += cycle_function

        return solver_program, pops, logbooks

    def optimize_relaxation_factors(self, expression, generations, min_level, max_level, base_program, storages):
        initial_weights = relaxation_factor_optimization.obtain_relaxation_factors(expression)
        relaxation_factor_optimization.set_relaxation_factors(expression, initial_weights)
        relaxation_factor_optimization.reset_status(expression)
        n = len(initial_weights)
        self.program_generator.initialize_code_generation(max_level)
        try:
            tmp = base_program + self.program_generator.generate_global_weights(n)
            cycle_function = self.program_generator.generate_cycle_function(expression, storages, min_level, max_level,
                                                                            max_level, use_global_weights=True)
            self.program_generator.generate_l3_file(tmp + cycle_function)
            best_individual = self._weight_optimizer.optimize(expression, n, generations, storages)
            best_weights = list(best_individual)
            spectral_radius, = best_individual.fitness.values
        except (KeyboardInterrupt, Exception) as e:
            self.program_generator.restore_files()
            raise e
        # print(f'Best weights: {best_weights}')
        # print(f'Best spectral radius: {spectral_radius}')
        self.program_generator.restore_files()
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
        ax2.set_ylabel("Number of operations", color="r")
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
        Optimizer.plot_multiobjective_data(gen, convergence_mins, runtime_mins, 'Minimum Spectral Radius', 'Minimum Estimated Number of Operations')

    @staticmethod
    def plot_average_fitness(logbook):
        gen = logbook.select("gen")
        convergence_avgs = logbook.chapters["convergence"].select("avg")
        runtime_avgs = logbook.chapters["runtime"].select("avg")
        Optimizer.plot_multiobjective_data(gen, convergence_avgs, runtime_avgs, 'Average Spectral Radius', 'Average Estimated Number of Operations')

    @staticmethod
    def plot_pareto_front(pop):
        import matplotlib.pyplot as plt
        import numpy
        pop.sort(key=lambda x: x.fitness.values)

        front = numpy.array([ind.fitness.values for ind in pop])
        plt.scatter(front[:, 0], front[:, 1], c="b")
        plt.xlabel("Spectral Radius")
        plt.ylabel("Number of Operations")
        plt.axis("tight")
        plt.show()

    @staticmethod
    def dump_population(pop, file_name):
        import pickle
        with open(file_name, 'wb') as file:
            pickle.dump(pop, file)



