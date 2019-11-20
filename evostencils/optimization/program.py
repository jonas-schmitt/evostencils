import numpy as np
import deap.base
from deap import gp, creator, tools, algorithms
import random
import pickle
import os.path
from evostencils.initialization import multigrid as multigrid_initialization
from evostencils.expressions import base, transformations, system
from evostencils.deap_extension import genGrow
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
                 program_generator=None, epsilon=1e-10, infinity=1e100, checkpoint_directory_path='./'):
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

    def _init_multi_objective_toolbox(self, pset):
        self._toolbox = deap.base.Toolbox()
        self._toolbox.register("expression", genGrow, pset=pset, min_height=10, max_height=50)
        self._toolbox.register("individual", tools.initIterate, creator.MultiObjectiveIndividual, self._toolbox.expression)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("evaluate", self.evaluate_multiple_objectives, pset=pset)
        self._toolbox.register("select", tools.selNSGA2, nd='log')
        self._toolbox.register("mate", gp.cxOnePoint)
        self._toolbox.register("expr_mut", genGrow, pset=pset, min_height=1, max_height=10)
        self._toolbox.register("mutate", gp.mutUniform, expr=self._toolbox.expr_mut, pset=pset)

    def _init_single_objective_toolbox(self, pset):
        self._toolbox = deap.base.Toolbox()
        self._toolbox.register("expression", genGrow, pset=pset, min_height=10, max_height=50)
        self._toolbox.register("individual", tools.initIterate, creator.SingleObjectiveIndividual, self._toolbox.expression)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("evaluate", self.evaluate_single_objective, pset=pset)
        self._toolbox.register("select", tools.selBest)
        self._toolbox.register("mate", gp.cxOnePoint)
        self._toolbox.register("expr_mut", genGrow, pset=pset, min_height=1, max_height=10)
        self._toolbox.register("mutate", gp.mutUniform, expr=self._toolbox.expr_mut, pset=pset)

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
                complexity = self.performance_evaluator.estimate_complexity(expression)
                return spectral_radius, complexity
            else:
                return spectral_radius, self.infinity

    def evaluate_single_objective(self, individual, pset):
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
                problem_size = min([reduce(lambda x, y: x * y, g.size) for g in expression.grid])
                complexity = self.performance_evaluator.estimate_complexity(expression) / problem_size
                rho = spectral_radius
                if rho < 1.0:
                    return math.log(self.epsilon) / math.log(rho) * complexity,
                else:
                    return rho * (100 * complexity),
            else:
                return spectral_radius,

    def gp_harm(self, initial_population_size, generations, crossover_probability, mutation_probability):
        self._toolbox.unregister('select')
        self._toolbox.register('select', tools.selTournament, tournsize=2)
        pop = self._toolbox.population(n=initial_population_size)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

        def mean(xs):
            avg = 0
            for x in xs:
                if isinstance(x, tuple):
                    x = x[0]
                else:
                    pass
                if x < self.infinity:
                    avg += x
            avg = avg / len(xs)
            return avg

        mstats.register("avg", mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)
        pop, log = gp.harm(pop, self._toolbox, crossover_probability, mutation_probability, generations, alpha=0.05,
                           beta=10, gamma=0.25, rho=0.9, stats=mstats, halloffame=hof, verbose=True)
        # print log
        return pop, log, hof

    def gp_mu_plus_lambda(self, initial_population_size, mu_, lambda_, generations, crossover_probability, mutation_probability):
        pop = self._toolbox.population(n=initial_population_size)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

        def mean(xs):
            avg = 0
            for x in xs:
                if isinstance(x, tuple):
                    x = x[0]
                else:
                    pass
                if x < self.infinity:
                    avg += x
            avg = avg / len(xs)
            return avg

        mstats.register("avg", mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)
        pop, log = algorithms.eaMuPlusLambda(pop, self._toolbox, mu_, lambda_, crossover_probability, mutation_probability,
                                             ngen=generations, stats=mstats, halloffame=hof, verbose=True)
        # print log
        return pop, log, hof

    def gp_mu_comma_lambda(self, initial_population_size, mu_, lambda_, generations, crossover_probability, mutation_probability):
        pop = self._toolbox.population(n=initial_population_size)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

        def mean(xs):
            avg = 0
            for x in xs:
                if isinstance(x, tuple):
                    x = x[0]
                else:
                    pass
                if x < self.infinity:
                    avg += x
            avg = avg / len(xs)
            return avg

        mstats.register("avg", mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)
        pop, log = algorithms.eaMuCommaLambda(pop, self._toolbox, mu_, lambda_, crossover_probability, mutation_probability,
                                              ngen=generations, stats=mstats, halloffame=hof, verbose=True)
        # print log
        return pop, log, hof

    def gp_nsgaII(self, initial_population_size, generations, mu_, lambda_, crossover_probability, mutation_probability,
                  min_level, max_level, program, solver, logbooks, checkpoint_frequency=5, checkpoint=None):
        print("Running NSGA-II")
        self._toolbox.unregister("select")
        self._toolbox.register("select", tools.selNSGA2, nd='log')
        random.seed()
        use_checkpoint = False
        if checkpoint is not None:
            if lambda_ == len(checkpoint.population):
                use_checkpoint = True
            else:
                print(f'Could not restart from checkpoint. Checkpoint population size is {len(checkpoint.population)} '
                      f'but the required size is {lambda_}.')
        if use_checkpoint:
            pop = checkpoint.population
            min_generation = checkpoint.generation
        else:
            pop = self._toolbox.population(n=initial_population_size)
            min_generation = 0
        max_generation = generations
        hof = tools.ParetoFront()

        stats_fit1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fit2 = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(convergence=stats_fit1, complexity=stats_fit2, size=stats_size)

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
        if use_checkpoint:
            logbook = logbooks[-1]
        else:
            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
            logbooks.append(logbook)

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
        logbook.record(gen=min_generation, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        # Begin the generational process
        for gen in range(min_generation + 1, max_generation + 1):
            # Vary the population
            offspring = tools.selTournamentDCD(pop, int(round(4 * lambda_) // 4))
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
                    transformations.invalidate_expression(solver)
                logbooks[-1] = logbook
                checkpoint = CheckPoint(min_level, max_level, gen, program, solver, pop, logbooks)
                try:
                    checkpoint.dump_to_file(f'{self._checkpoint_directory_path}/checkpoint.p')
                except (pickle.PickleError, TypeError) as e:
                    print(e)
                    print('Skipping checkpoint')
            # Select the next generation population
            pop = toolbox.select(pop + offspring, mu_)
            record = mstats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

        return pop, logbook, hof

    def gp_nsgaIII(self, initial_population_size, generations, mu_, lambda_, crossover_probability, mutation_probability,
                   min_level, max_level, program, solver, logbooks, checkpoint_frequency=5, checkpoint=None):
        print("Running NSGA-III")
        self._toolbox.unregister("select")
        ref_points = tools.uniform_reference_points(2, 12)
        self._toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
        return self.gp_multi_objective(initial_population_size, generations, mu_, lambda_, crossover_probability,
                                       mutation_probability, min_level, max_level, program, solver, logbooks,
                                       checkpoint_frequency, checkpoint)

    def gp_speaII(self, initial_population_size, generations, mu_, lambda_, crossover_probability, mutation_probability,
                  min_level, max_level, program, solver, logbooks, checkpoint_frequency=5, checkpoint=None):
        print("Running SPEA-II")
        self._toolbox.unregister("select")
        self._toolbox.register("select", tools.selSPEA2)
        return self.gp_multi_objective(initial_population_size, generations, mu_, lambda_, crossover_probability,
                                       mutation_probability, min_level, max_level, program, solver, logbooks,
                                       checkpoint_frequency, checkpoint)

    def gp_multi_objective(self, initial_population_size, generations, mu_, lambda_, crossover_probability,
                           mutation_probability, min_level, max_level, program, solver, logbooks,
                           checkpoint_frequency=5, checkpoint=None):
        random.seed()
        use_checkpoint = False
        if checkpoint is not None:
            if lambda_ == len(checkpoint.population):
                use_checkpoint = True
            else:
                print(f'Could not restart from checkpoint. Checkpoint population size is {len(checkpoint.population)} '
                      f'but the required size is {lambda_}.')
        if use_checkpoint:
            pop = checkpoint.population
            min_generation = checkpoint.generation
        else:
            pop = self._toolbox.population(n=initial_population_size)
            min_generation = 0
        max_generation = generations
        hof = tools.ParetoFront()

        stats_fit1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fit2 = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(convergence=stats_fit1, complexity=stats_fit2, size=stats_size)

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
        if use_checkpoint:
            logbook = logbooks[-1]
        else:
            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
            logbooks.append(logbook)

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
        logbook.record(gen=min_generation, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        # Begin the generational process
        for gen in range(min_generation + 1, max_generation + 1):
            # Vary the population
            offspring = algorithms.varOr(pop, toolbox, lambda_, crossover_probability, mutation_probability)
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            hof.update(pop)
            if gen % checkpoint_frequency == 0:
                if solver is not None:
                    transformations.invalidate_expression(solver)
                logbooks[-1] = logbook
                checkpoint = CheckPoint(min_level, max_level, gen, program, solver, pop, logbooks)
                try:
                    checkpoint.dump_to_file(f'{self._checkpoint_directory_path}/checkpoint.p')
                except (pickle.PickleError, TypeError) as e:
                    print(e)
                    print('Skipping checkpoint')
            # Select the next generation population
            pop = toolbox.select(pop + offspring, mu_)
            record = mstats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

        return pop, logbook, hof

    def evolutionary_single_objective_optimization(self, levels_per_run=2, gp_mu=100, gp_lambda=100, gp_generations=100,
                                                   gp_crossover_probability=0.7, gp_mutation_probability=0.3,
                                                   es_generations=100, required_convergence=0.1, maximum_block_size=2):

        levels = self.max_level - self.min_level
        approximations = [self.approximation]
        right_hand_sides = [self.rhs]
        for i in range(1, levels + 1):
            approximations.append(system.get_coarse_approximation(approximations[-1], self.coarsening_factors))
            right_hand_sides.append(system.get_coarse_rhs(right_hand_sides[-1], self.coarsening_factors))
        best_expression = None
        solver_program = ""
        pops = []
        logbooks = []
        storages = self._program_generator.generate_storage(self.min_level, self.max_level, self.finest_grid)
        for i in range(levels - levels_per_run, -1, -levels_per_run):
            min_level = self.max_level - (i + levels_per_run - 1)
            max_level = self.max_level - i
            approximation = approximations[i]

            self._convergence_evaluator.reinitialize_lfa_grids(approximation.grid)

            rhs = right_hand_sides[i]
            pset = multigrid_initialization.generate_primitive_set(approximation, rhs, self.dimension, self.coarsening_factors,
                                                                   max_level, self.equations, self.operators, self.fields,
                                                                   maximum_block_size=maximum_block_size,
                                                                   coarse_grid_solver_expression=best_expression,
                                                                   depth=levels_per_run, LevelFinishedType=self._FinishedType,
                                                                   LevelNotFinishedType=self._NotFinishedType)
            self._init_single_objective_toolbox(pset)

            # pop, log, hof = self.gp_harm(gp_mu, gp_generations, gp_crossover_probability, gp_mutation_probability)
            pop, log, hof = self.gp_mu_plus_lambda(10 * gp_mu, gp_mu, gp_lambda, gp_generations, gp_crossover_probability,
                                                   gp_mutation_probability)
            # pop, log, hof = self.gp_mu_comma_lambda(10 * gp_mu, gp_mu, gp_lambda, gp_generations, gp_crossover_probability,
            #                                         gp_mutation_probability)

            pops.append(pop)
            pop = sorted(pop, key=lambda ind: ind.fitness.values[0])
            best_time = self.infinity
            self.program_generator._counter = 0
            self.program_generator._average_generation_time = 0
            self.program_generator.initialize_code_generation(max_level)
            count = 0
            try:
                for j in range(0, len(pop)):
                    if j < len(pop) - 1 and abs(pop[j].fitness.values[0] - pop[j + 1].fitness.values[0]) < self.epsilon:
                        continue
                    count += 1
                    if count >= 100:
                        break
                    individual = pop[j]
                    expression = self.compile_individual(individual, pset)[0]

                    time, convergence_factor = \
                        self._program_generator.generate_and_evaluate(expression, storages, min_level, max_level,
                                                                      solver_program, number_of_samples=100)
                    print(f'Time: {time}, Measured convergence factor: {convergence_factor}')
                    individual.fitness.values = (time,)
                    if time < best_time and ((i == 0 and convergence_factor < 0.9)
                                             or convergence_factor < required_convergence):
                        best_expression = expression
                        best_time = time

            except (KeyboardInterrupt, Exception) as e:
                self.program_generator.restore_files()
                raise e
            self.program_generator.restore_files()

            relaxation_factors, _ = self.optimize_relaxation_factors(best_expression, es_generations, min_level, max_level, solver_program, storages)
            relaxation_factor_optimization.set_relaxation_factors(best_expression, relaxation_factors)

            cycle_function = self.program_generator.generate_cycle_function(best_expression, storages, min_level,
                                                                             max_level, self.max_level)
            solver_program += cycle_function

        return solver_program, pops, logbooks

    def evolutionary_multi_objective_optimization(self, levels_per_run=1, gp_mu=100, gp_lambda=100, gp_generations=100, gp_crossover_probability=0.7,
                                                  gp_mutation_probability=0.3, es_generations=100, required_convergence=0.5,
                                                  restart_from_checkpoint=False, maximum_block_size=2):

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
                elif min_level > checkpoint.min_level:
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
            self._init_multi_objective_toolbox(pset)
            tmp = None
            if pass_checkpoint:
                tmp = checkpoint

            pop, log, hof = self.gp_nsgaII(10 * gp_mu, gp_generations, gp_mu, gp_lambda, gp_crossover_probability,
                                           gp_mutation_probability, min_level, max_level, solver_program, best_expression, logbooks,
                                           checkpoint_frequency=5, checkpoint=tmp)

            pops.append(pop)

            hof = sorted(hof, key=lambda ind: ind.fitness.values[1])
            best_time = self.infinity
            best_individual = hof[0]
            self.program_generator._counter = 0
            self.program_generator._average_generation_time = 0
            self.program_generator.initialize_code_generation(max_level)
            try:
                for j in range(0, min(100, len(hof))):
                    if j < len(hof) - 1 and abs(hof[j].fitness.values[0] - hof[j + 1].fitness.values[0]) < self.epsilon and \
                            abs(hof[j].fitness.values[1] - hof[j + 1].fitness.values[1] < self.epsilon):
                        continue
                    individual = hof[j]
                    expression = self.compile_individual(individual, pset)[0]

                    time, convergence_factor = \
                        self._program_generator.generate_and_evaluate(expression, storages, min_level, max_level,
                                                                      solver_program, number_of_samples=100)
                    print(f'Time: {time}, Estimated convergence factor: {individual.fitness.values[0]}, '
                          f'Measured convergence factor: {convergence_factor}')
                    individual.fitness.values = (convergence_factor, individual.fitness.values[1])
                    if time < best_time and ((i == 0 and convergence_factor < 0.9)
                                             or convergence_factor < required_convergence):
                        best_individual = individual
                        best_expression = expression
                        best_time = time

            except (KeyboardInterrupt, Exception) as e:
                self.program_generator.restore_files()
                raise e
            self.program_generator.restore_files()
            if best_expression is None:
                raise RuntimeError("Optimization failed")
            best_convergence_factor = best_individual.fitness.values[0]
            print(f"Best individual: ({best_convergence_factor}), ({best_individual.fitness.values[1]})")
            relaxation_factors, _ = self.optimize_relaxation_factors(best_expression, es_generations, min_level, max_level, solver_program, storages)
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
    def plot_multiobjective_data(generations, convergence_data, complexity_data, label1, label2):
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
        line2 = ax2.plot(generations, complexity_data, "r-", label=f"{label2}")
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
        complexity_mins = logbook.chapters["complexity"].select("min")
        Optimizer.plot_multiobjective_data(gen, convergence_mins, complexity_mins, 'Minimum Spectral Radius', 'Minimum Estimated Number of Operations')

    @staticmethod
    def plot_average_fitness(logbook):
        gen = logbook.select("gen")
        convergence_avgs = logbook.chapters["convergence"].select("avg")
        complexity_avgs = logbook.chapters["complexity"].select("avg")
        Optimizer.plot_multiobjective_data(gen, convergence_avgs, complexity_avgs, 'Average Spectral Radius', 'Average Estimated Number of Operations')

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



