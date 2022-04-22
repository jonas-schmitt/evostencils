import deap.base
from deap import gp, creator, tools
import random
import pickle
import os.path
from evostencils.grammar import multigrid as multigrid_initialization
from evostencils.ir import base, transformations, system
from evostencils.genetic_programming import genGrow, mutNodeReplacement, mutInsert, select_unique_best
from evostencils.types import level_control
import math
import numpy as np
import time
import os
# from mpi4py import MPI


def flatten(lst: list):
    return [item for sublist in lst for item in sublist]


class do_nothing(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *_):
        pass


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
                 program_generator, convergence_evaluator=None, performance_evaluator=None,
                 mpi_comm=None, mpi_rank=0, number_of_mpi_processes=1,
                 epsilon=1e-12, infinity=1e100, checkpoint_directory_path='./'):
        assert program_generator is not None, "At least a program generator must be available"
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
        self._total_number_of_evaluations = 0
        self._failed_evaluations = 0
        self._mpi_comm = mpi_comm
        self._mpi_rank = mpi_rank
        self._number_of_mpi_processes = number_of_mpi_processes
        self._individual_cache = {}
        self._individual_cache_size = 100000
        self._individual_cache_hits = 0
        self._individual_cache_misses = 0
        self._timeout_counter_limit = 10000
        self._total_evaluation_time = 0
        self._average_time_to_convergence = infinity

    def reinitialize_code_generation(self, min_level, max_level, program, evaluation_function, evaluation_samples=3,
                                     pde_parameter_values=None):
        if pde_parameter_values is None:
            pde_parameter_values = {}
        self._average_time_to_convergence = self.infinity
        self.program_generator.reinitialize(min_level, max_level, pde_parameter_values)
        program_generator = self.program_generator
        dimension = program_generator.dimension
        finest_grid = program_generator.finest_grid
        coarsening_factor = program_generator.coarsening_factor
        min_level = program_generator.min_level
        max_level = program_generator.max_level
        equations = program_generator.equations
        operators = program_generator.operators
        fields = program_generator.fields
        solution_entries = [base.Approximation(f.name, g) for f, g in zip(fields, finest_grid)]
        approximation = system.Approximation('x', solution_entries)
        rhs_entries = [base.RightHandSide(eq.rhs_name, g) for eq, g in zip(equations, finest_grid)]
        rhs = system.RightHandSide('b', rhs_entries)
        self.clear_individual_cache()
        maximum_block_size = 8
        levels_per_run = max_level - min_level
        pset, _ = \
            multigrid_initialization.generate_primitive_set(approximation, rhs, dimension,
                                                            coarsening_factor, max_level, equations,
                                                            operators, fields,
                                                            maximum_local_system_size=maximum_block_size,
                                                            depth=levels_per_run,
                                                            LevelFinishedType=self._FinishedType,
                                                            LevelNotFinishedType=self._NotFinishedType)
        self.program_generator.initialize_code_generation(min_level, max_level)
        storages = self._program_generator.generate_storage(min_level, max_level, finest_grid)
        self.toolbox.register('evaluate', evaluation_function, pset=pset,
                              storages=storages, min_level=min_level, max_level=max_level,
                              solver_program=program, evaluation_samples=evaluation_samples,
                              pde_parameter_values=pde_parameter_values)

    @staticmethod
    def _init_creator():
        creator.create("MultiObjectiveFitness", deap.base.Fitness, weights=(-1.0, -1.0))
        creator.create("MultiObjectiveIndividual", gp.PrimitiveTree, fitness=creator.MultiObjectiveFitness)
        creator.create("SingleObjectiveFitness", deap.base.Fitness, weights=(-1.0,))
        creator.create("SingleObjectiveIndividual", gp.PrimitiveTree, fitness=creator.SingleObjectiveFitness)

    def _init_toolbox(self, pset, node_replacement_probability=1.0/3.0):
        self._toolbox = deap.base.Toolbox()
        self._toolbox.register("expression", genGrow, pset=pset, min_height=0, max_height=50)
        self._toolbox.register("mate", gp.cxOnePoint)

        def mutate(individual, pset_):
            # Use two different mutation operators
            operator_choice = random.random()
            if operator_choice < node_replacement_probability:
                return mutNodeReplacement(individual, pset_)
            else:
                return mutInsert(individual, 0, 10, pset_)

        self._toolbox.register("mutate", mutate, pset_=pset)

    def _init_multi_objective_toolbox(self):
        self._toolbox.register("individual", tools.initIterate, creator.MultiObjectiveIndividual,
                               self._toolbox.expression)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)

    def _init_single_objective_toolbox(self):
        self._toolbox.register("individual", tools.initIterate, creator.SingleObjectiveIndividual,
                               self._toolbox.expression)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)

    @property
    def toolbox(self):
        return self._toolbox

    @property
    def individual_cache(self):
        return self._individual_cache

    def clear_individual_cache(self):
        self.individual_cache.clear()

    def add_individual_to_cache(self, individual, values):
        if len(self.individual_cache) < self._individual_cache_size:
            self.individual_cache[str(individual)] = values

    def individual_in_cache(self, individual):
        tmp = str(individual) in self.individual_cache
        if tmp:
            self._individual_cache_hits += 1
        else:
            self._individual_cache_misses += 1
        return tmp

    def get_cached_fitness(self, individual):
        return self.individual_cache[str(individual)]

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

    @property
    def mpi_rank(self):
        return self._mpi_rank

    @property
    def mpi_comm(self):
        return self._mpi_comm

    @property
    def number_of_mpi_processes(self):
        return self._number_of_mpi_processes

    def is_root(self):
        return self.mpi_rank == 0

    @property
    def total_evaluation_time(self):
        return self._total_evaluation_time

    def allgather(self, data):
        if self.mpi_comm is None:
            return data
        else:
            return flatten(self.mpi_comm.allgather(data))

    def gather(self, data):
        if self.mpi_comm is None:
            return data
        else:
            tmp = self.mpi_comm.gather(data)
            if self.is_root():
                return flatten(tmp)
            else:
                return tmp

    def allreduce(self, variable):
        if self.mpi_comm is None:
            return variable
        else:
            from mpi4py import MPI
            return self.mpi_comm.allreduce(variable, op=MPI.SUM)

    def barrier(self):
        if self.mpi_comm is not None:
            self.mpi_comm.barrier()

    def generate_individual(self):
        return self._toolbox.individual()

    @staticmethod
    def compile_individual(individual, pset):
        return gp.compile(individual, pset)

    def estimate_single_objective(self, individual, pset):
        self._total_number_of_evaluations += 1
        if self.individual_in_cache(individual):
            return self.get_cached_fitness(individual)
        with suppress_output():
            try:
                expression1, expression2 = self.compile_individual(individual, pset)
            except MemoryError:
                self._failed_evaluations += 1
                values = self.infinity,
                self.add_individual_to_cache(individual, values)
                return values

        expression = expression1
        with suppress_output():
            spectral_radius = self.convergence_evaluator.compute_spectral_radius(expression)

        if spectral_radius == 0.0 or math.isnan(spectral_radius) \
                or math.isinf(spectral_radius) or np.isinf(spectral_radius) or np.isnan(spectral_radius):
            values = self.infinity,
            self.add_individual_to_cache(individual, values)
            return values
        else:
            if self.performance_evaluator is None:
                values = spectral_radius,
                self.add_individual_to_cache(individual, values)
                return values
            elif spectral_radius < 1:
                runtime = self.performance_evaluator.estimate_runtime(expression) * 1e3
                values = math.log(self.epsilon) / math.log(spectral_radius) * runtime,
                self.add_individual_to_cache(individual, values)
                return values
            else:
                values = spectral_radius * self.infinity**0.25,
                self.add_individual_to_cache(individual, values)
                return values

    def estimate_multiple_objectives(self, individual, pset):
        if self.individual_in_cache(individual):
            return self.get_cached_fitness(individual)
        self._total_number_of_evaluations += 1
        with suppress_output():
            # with do_nothing():
            try:
                expression1, expression2 = self.compile_individual(individual, pset)
            except MemoryError:
                self._failed_evaluations += 1
                values = self.infinity, self.infinity
                self.add_individual_to_cache(individual, values)
                return values

        expression = expression1
        with suppress_output():
            spectral_radius = self.convergence_evaluator.compute_spectral_radius(expression)

        if spectral_radius == 0.0 or math.isnan(spectral_radius) \
                or math.isinf(spectral_radius) or np.isinf(spectral_radius) or np.isnan(spectral_radius):
            self._failed_evaluations += 1
            values = self.infinity, self.infinity
            self.add_individual_to_cache(individual, values)
            return values
        else:
            runtime = self.performance_evaluator.estimate_runtime(expression) * 1e3
            values = spectral_radius, runtime
            self.add_individual_to_cache(individual, values)
            return values

    def evaluate_single_objective(self, individual, pset, storages, min_level, max_level, solver_program,
                                  evaluation_samples=3, pde_parameter_values=None):
        if pde_parameter_values is None:
            pde_parameter_values = {}
        if len(individual) > 150:
            return self.infinity,
        if self.individual_in_cache(individual):
            return self.get_cached_fitness(individual)
        with suppress_output():
            # with do_nothing():
            try:
                expression1, expression2 = self.compile_individual(individual, pset)
            except MemoryError:
                print("Memory Error", flush=True)
                self._failed_evaluations += 1
                fitness = self.infinity,
                self.add_individual_to_cache(individual, fitness)
                return fitness
            expression = expression1
            start = time.time()
            average_time_to_convergence, average_convergence_factor, average_number_of_iterations = \
                self._program_generator.generate_and_evaluate(expression, storages, min_level, max_level, solver_program,
                                                              infinity=self.infinity, evaluation_samples=evaluation_samples,
                                                              global_variable_values=pde_parameter_values)
            end = time.time()
            self._total_number_of_evaluations += 1
            self._total_evaluation_time += end - start
            fitness = average_time_to_convergence,
            if average_number_of_iterations >= self.infinity:
                fitness = average_convergence_factor**0.5 * average_number_of_iterations**0.5,
            self.add_individual_to_cache(individual, fitness)
            return fitness

    def evaluate_multiple_objectives(self, individual, pset, storages, min_level, max_level, solver_program,
                                     evaluation_samples=3, pde_parameter_values=None):
        if pde_parameter_values is None:
            pde_parameter_values = {}
        if len(individual) > 150:
            return self.infinity, self.infinity
        if self.individual_in_cache(individual):
            return self.get_cached_fitness(individual)
        with suppress_output():
        # with do_nothing():
            try:
                expression1, expression2 = self.compile_individual(individual, pset)
            except MemoryError:
                self._failed_evaluations += 1
                fitness = self.infinity, self.infinity
                self.add_individual_to_cache(individual, fitness)
                return fitness
            expression = expression1
            start = time.time()
            average_time_to_convergence, average_convergence_factor, average_number_of_iterations = \
                self._program_generator.generate_and_evaluate(expression, storages, min_level, max_level,
                                                              solver_program,
                                                              infinity=self.infinity,
                                                              evaluation_samples=evaluation_samples,
                                                              global_variable_values=pde_parameter_values)
            end = time.time()
            self._total_number_of_evaluations += 1
            self._total_evaluation_time += end - start
            # Use number of iteration
            # fitness = average_number_of_iterations, average_time_to_convergence / average_number_of_iterations
            fitness = average_convergence_factor, average_time_to_convergence / average_number_of_iterations
            if average_number_of_iterations >= self.infinity:
                fitness = average_convergence_factor, self.infinity
            self.add_individual_to_cache(individual, fitness)
            return fitness

    def ea_mu_plus_lambda(self, initial_population_size, generations, generalization_interval, mu_, lambda_,
                          crossover_probability, mutation_probability, min_level, max_level,
                          program, solver, evaluation_samples, logbooks, pde_parameter_values, checkpoint_frequency,
                          checkpoint, mstats, hof, use_random_search):

        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        self._average_time_to_convergence = self.infinity

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
            initial_population_size_per_process = int(math.ceil(initial_population_size / self.number_of_mpi_processes))
            population = self.toolbox.population(n=initial_population_size_per_process)
            min_generation = 0
        max_generation = generations

        if use_checkpoint:
            logbook = logbooks[-1]
        else:
            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
            logbooks.append(logbook)

        invalid_ind = [ind for ind in population]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population = self.allgather(population)
        population = self.toolbox.select(population, mu_)
        hof.update(population)
        if self.mpi_comm is not None and self.number_of_mpi_processes > 1:
            individual_caches = self.mpi_comm.allgather(self.individual_cache)
            for i, cache in enumerate(individual_caches):
                if i != self.mpi_rank:
                    self.individual_cache.update(cache)
        record = mstats.compile(population) if mstats is not None else {}
        logbook.record(gen=min_generation, nevals=len(population), **record)
        if self.is_root():
            print(logbook.stream, flush=True)
        # Begin the generational process
        count = 0
        evaluation_min_level = min_level
        evaluation_max_level = max_level
        level_offset = 0
        for gen in range(min_generation + 1, max_generation + 1):
            self._total_evaluation_time = 0.0
            self._total_number_of_evaluations = 0.0
            if count >= generalization_interval:
                level_offset += 1
                evaluation_min_level = min_level + level_offset
                evaluation_max_level = max_level + level_offset
                next_pde_parameter_values = {}
                for key, values in pde_parameter_values.items():
                    assert level_offset < len(values), 'Too few parameter values provided'
                    next_pde_parameter_values[key] = values[level_offset]
                count = 0
                if self.is_root():
                    print("Increasing problem size", flush=True)
                if len(population[0].fitness.values) == 2:
                    self.reinitialize_code_generation(evaluation_min_level, evaluation_max_level, program,
                                                      self.evaluate_multiple_objectives, evaluation_samples=evaluation_samples, pde_parameter_values=next_pde_parameter_values)
                else:
                    self.reinitialize_code_generation(evaluation_min_level, evaluation_max_level, program,
                                                      self.evaluate_single_objective, evaluation_samples=evaluation_samples, pde_parameter_values=next_pde_parameter_values)
                hof.clear()
                fitnesses = [(i, self.toolbox.evaluate(ind)) for i, ind in enumerate(population)
                             if i % self.number_of_mpi_processes == self.mpi_rank]
                fitnesses = self.allgather(fitnesses)
                for i, values in fitnesses:
                    population[i].fitness.values = values
                population = self.toolbox.select(population, mu_)
                hof.update(population)

            if use_random_search:
                offspring = self.toolbox.population(n=lambda_)
            else:
                number_of_parents = lambda_
                if number_of_parents % 2 == 1:
                    number_of_parents += 1
                selected = self.toolbox.select_for_mating(population, number_of_parents)
                parents = [self.toolbox.clone(ind) for ind in selected]
                offspring = []
                for ind1, ind2 in zip(parents[::2], parents[1::2]):
                    child1 = None
                    child2 = None
                    tries = 0
                    while tries < 10 and (child1 is None or len(child1) > 150 or self.individual_in_cache(child1) or
                                          child2 is None or len(child2) > 150 or self.individual_in_cache(child2)):
                        operator_choice = random.random()
                        if operator_choice < crossover_probability:
                            child1, child2 = self.toolbox.mate(ind1, ind2)
                        elif operator_choice < crossover_probability + mutation_probability + self.epsilon:
                            child1, = self.toolbox.mutate(ind1)
                            child2, = self.toolbox.mutate(ind2)
                        else:
                            child1 = ind1
                            child2 = ind2
                        tries += 1
                    del child1.fitness.values, child2.fitness.values
                    offspring.append(child1)
                    if len(offspring) == lambda_:
                        break
                    offspring.append(child2)
                    if len(offspring) == lambda_:
                        break

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            self._total_evaluation_time = self.allreduce(self.total_evaluation_time)
            self._total_number_of_evaluations = self.allreduce(self._total_number_of_evaluations)
            offspring = self.allgather(offspring)
            hof.update(offspring)

            if self.mpi_comm is not None and self.number_of_mpi_processes > 1:
                for ind in offspring:
                    if not self.individual_in_cache(ind):
                        self.add_individual_to_cache(ind, ind.fitness.values)

            if gen % checkpoint_frequency == 0:
                if solver is not None:
                    transformations.invalidate_expression(solver)
                logbooks[-1] = logbook
                checkpoint = CheckPoint(min_level, max_level, gen, program, solver, population, logbooks)
                try:
                    if not os.path.exists(self._checkpoint_directory_path):
                        os.makedirs(self._checkpoint_directory_path)
                    checkpoint.dump_to_file(f'{self._checkpoint_directory_path}/checkpoint.p')
                except (pickle.PickleError, TypeError, FileNotFoundError) as e:
                    print(e, flush=True)
                    print(f'Skipping checkpoint on process with rank {self.mpi_rank}', flush=True)
            # Select the next generation population
            elitism = 1.0
            assert int(mu_*elitism) + len(offspring) >= mu_
            population = self.toolbox.select(population, int(mu_*elitism))
            population = self.toolbox.select(population + offspring, mu_)
            total_time_to_convergence = 0
            if len(population[0].fitness.values) == 1:
                for ind in population:
                    total_time_to_convergence += ind.fitness.values[0]
            else:
                for ind in population:
                    total_time_to_convergence += ind.fitness.values[0] * ind.fitness.values[1]
            self._average_time_to_convergence = total_time_to_convergence / len(population)
            count += 1
            record = mstats.compile(population)
            # Update the statistics with the new population
            logbook.record(gen=gen, nevals=len(offspring), **record)
            if self.is_root():
                print(logbook.stream, flush=True)
        hof.update(population)
        if self.is_root():
            print("Optimization finished", flush=True)

        return population, logbook, hof, evaluation_min_level, evaluation_max_level

    def SOGP(self, pset, initial_population_size, generations, generalization_interval, mu_, lambda_,
             crossover_probability, mutation_probability, min_level, max_level,
             program, storages, solver, evaluation_samples, logbooks, use_random_search=False,
             model_based_estimation=False, pde_parameter_values=None, checkpoint_frequency=2, checkpoint=None):
        if pde_parameter_values is None:
            pde_parameter_values = {}
        elif model_based_estimation and self.is_root():
            print("Warning: Parametrization not supported in model-based estimation")
        if self.is_root():
            msg = "Running Single-Objective"
            if use_random_search:
                msg += " Random Search"
            else:
                msg += " Genetic Programming"
            if model_based_estimation:
                print(msg, "with Model-Based Estimation")
            else:
                print(msg, "with Code Generation-Based Evaluation")
        self._init_single_objective_toolbox()
        self._toolbox.register("select", select_unique_best)
        self._toolbox.register("select_for_mating", tools.selTournament, tournsize=2)
        if model_based_estimation:
            self._toolbox.register('evaluate', self.estimate_single_objective, pset=pset)
        else:
            initial_pde_parameter_values = {}
            for key, values in pde_parameter_values.items():
                initial_pde_parameter_values[key] = values[0]
            self._toolbox.register('evaluate', self.evaluate_single_objective, pset=pset,
                                   storages=storages, min_level=min_level, max_level=max_level, solver_program=program,
                                   evaluation_samples=evaluation_samples,
                                   pde_parameter_values=initial_pde_parameter_values)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

        hof = tools.HallOfFame(2 * mu_, similar=lambda a, b: str(a) == str(b))
        return self.ea_mu_plus_lambda(initial_population_size, generations, generalization_interval, mu_, lambda_,
                                      crossover_probability, mutation_probability, min_level, max_level,
                                      program, solver, evaluation_samples, logbooks, pde_parameter_values,
                                      checkpoint_frequency, checkpoint, mstats, hof, use_random_search)

    def NSGAII(self, pset, initial_population_size, generations, generalization_interval, mu_, lambda_,
               crossover_probability, mutation_probability, min_level, max_level,
               program, storages, solver, evaluation_samples, logbooks, use_random_search=False,
               model_based_estimation=False, pde_parameter_values=None, checkpoint_frequency=2, checkpoint=None):

        if pde_parameter_values is None:
            pde_parameter_values = {}
        elif model_based_estimation and self.is_root():
            print("Warning: Parametrization not supported in model-based estimation")
        if self.is_root():
            msg = "Running NSGA-II Multi-Objective"
            if use_random_search:
                msg += " Random Search"
            else:
                msg += " Genetic Programming"
            if model_based_estimation:
                print(msg, "with Model-Based Estimation")
            else:
                print(msg, "with Code Generation-Based Evaluation")
        self._init_multi_objective_toolbox()
        self._toolbox.register("select", tools.selNSGA2)

        def select_for_mating(individuals, k):
            if k % 4 > 0:
                k = k + (4 - k % 4)
            return tools.selTournamentDCD(individuals, k)

        self._toolbox.register("select_for_mating", select_for_mating)
        if model_based_estimation:
            self._toolbox.register('evaluate', self.estimate_multiple_objectives, pset=pset)
        else:
            initial_pde_parameter_values = {}
            for key, values in pde_parameter_values.items():
                initial_pde_parameter_values[key] = values[0]
            self._toolbox.register('evaluate', self.evaluate_multiple_objectives, pset=pset,
                                   storages=storages, min_level=min_level, max_level=max_level,
                                   solver_program=program, evaluation_samples=evaluation_samples,
                                   pde_parameter_values=initial_pde_parameter_values)

        stats_fit1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fit2 = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(convergence_factor=stats_fit1, execution_time=stats_fit2, size=stats_size)

        hof = tools.ParetoFront(similar=lambda a, b: str(a) == str(b))

        return self.ea_mu_plus_lambda(initial_population_size, generations, generalization_interval, mu_, lambda_,
                                      crossover_probability, mutation_probability, min_level, max_level,
                                      program, solver, evaluation_samples, logbooks, pde_parameter_values,
                                      checkpoint_frequency, checkpoint, mstats, hof, use_random_search)

    def NSGAIII(self, pset, initial_population_size, generations, generalization_interval, mu_, lambda_,
                crossover_probability, mutation_probability, min_level, max_level,
                program, storages, solver, evaluation_samples, logbooks, use_random_search=False,
                model_based_estimation=False, pde_parameter_values=None, checkpoint_frequency=2, checkpoint=None):
        if pde_parameter_values is None:
            pde_parameter_values = {}
        elif model_based_estimation and self.is_root():
            print("Warning: Parametrization not supported in model-based estimation")

        if self.is_root():
            msg = "Running NSGA-III Multi-Objective"
            if use_random_search:
                msg += " Random Search"
            else:
                msg += " Genetic Programming"
            if model_based_estimation:
                print(msg, "with Model-Based Estimation")
            else:
                print(msg, "with Code Generation-Based Evaluation")

        self._init_multi_objective_toolbox()
        H = mu_
        reference_points = tools.uniform_reference_points(2, H)
        if H % 4 > 0:
            mu_ = H + (4 - H % 4)
        self._toolbox.register("select", tools.selNSGA3, ref_points=reference_points)
        self._toolbox.register("select_for_mating", tools.selRandom)
        if model_based_estimation:
            self._toolbox.register('evaluate', self.estimate_multiple_objectives, pset=pset)
        else:
            initial_pde_parameter_values = {}
            for key, values in pde_parameter_values.items():
                initial_pde_parameter_values[key] = values[0]
            self._toolbox.register('evaluate', self.evaluate_multiple_objectives, pset=pset,
                                   storages=storages, min_level=min_level, max_level=max_level,
                                   solver_program=program, evaluation_samples=evaluation_samples,
                                   pde_parameter_values=initial_pde_parameter_values)

        stats_fit1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fit2 = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(convergence_factor=stats_fit1, execution_time=stats_fit2, size=stats_size)

        hof = tools.ParetoFront(similar=lambda a, b: str(a) == str(b))

        return self.ea_mu_plus_lambda(initial_population_size, generations, generalization_interval, mu_, lambda_,
                                      crossover_probability, mutation_probability, min_level, max_level,
                                      program, solver, evaluation_samples, logbooks, pde_parameter_values,
                                      checkpoint_frequency, checkpoint, mstats, hof, use_random_search)

    def evolutionary_optimization(self, mu_=128, lambda_=128, population_initialization_factor=4, generations=150,
                                  generalization_interval=50, crossover_probability=0.7, mutation_probability=0.3,
                                  node_replacement_probability=0.1, optimization_method=None, use_random_search=False,
                                  levels_per_run=None, evaluation_samples=3, continue_from_checkpoint=False,
                                  maximum_local_system_size=8, model_based_estimation=False, pde_parameter_values=None):
        levels = self.max_level - self.min_level
        if levels_per_run is None:
            levels_per_run = levels
        if levels_per_run < levels and generalization_interval < generations:
            print("Warning: Stepwise generalization only supported for single-stage optimizations")
            print("Adapting generalization interval accordingly...")
            generalization_interval = generations
        approximations = [self.approximation]
        right_hand_sides = [self.rhs]
        for i in range(1, levels + 1):
            approximations.append(system.get_coarse_approximation(approximations[-1], self.coarsening_factors))
            right_hand_sides.append(system.get_coarse_rhs(right_hand_sides[-1], self.coarsening_factors))
        best_expression = None
        best_individual = None
        checkpoint = None
        checkpoint_file_path = f'{self._checkpoint_directory_path}/checkpoint.p'
        solver_program = ""
        if continue_from_checkpoint and os.path.isfile(checkpoint_file_path):
            try:
                checkpoint = load_checkpoint_from_file(checkpoint_file_path)
                solver_program = checkpoint.program
            except pickle.PickleError as _:
                continue_from_checkpoint = False
        else:
            continue_from_checkpoint = False
        pops = []
        logbooks = []
        hofs = []
        storages = self._program_generator.generate_storage(self.min_level, self.max_level, self.finest_grid)
        FAS = False
        if self._program_generator.uses_FAS:
            FAS = True
        for i in range(0, levels, levels_per_run):
            min_level = self.max_level - (i + levels_per_run)
            max_level = self.max_level - i
            pass_checkpoint = False
            if continue_from_checkpoint:
                if min_level == checkpoint.min_level and max_level == checkpoint.max_level:
                    best_expression = checkpoint.solver
                    pass_checkpoint = True
                    logbooks = checkpoint.logbooks
                elif min_level < checkpoint.min_level:
                    continue
            approximation = approximations[i]

            if model_based_estimation:
                self.convergence_evaluator.reinitialize_lfa_grids(approximation.grid)
            if model_based_estimation and i > 0 and self.performance_evaluator is not None:
                self.performance_evaluator.set_runtime_of_coarse_grid_solver(0.0)

            rhs = right_hand_sides[i]
            enable_partitioning = True
            if model_based_estimation:
                print("Warning: Smoother partitioning not supported with model-based estimation")
                enable_partitioning = False
            pset, terminal_list = \
                multigrid_initialization.generate_primitive_set(approximation, rhs, self.dimension,
                                                                self.coarsening_factors, max_level, self.equations,
                                                                self.operators, self.fields,
                                                                enable_partitioning=enable_partitioning,
                                                                maximum_local_system_size=maximum_local_system_size,
                                                                depth=levels_per_run,
                                                                LevelFinishedType=self._FinishedType,
                                                                LevelNotFinishedType=self._NotFinishedType,
                                                                FAS=FAS)
            self._init_toolbox(pset, node_replacement_probability)
            tmp = None
            if pass_checkpoint:
                tmp = checkpoint
            initial_population_size = population_initialization_factor * mu_

            self.program_generator._counter = 0
            self.program_generator._average_generation_time = 0

            self.program_generator.initialize_code_generation(self.min_level, self.max_level)
            if optimization_method is None:
                optimization_method = self.NSGAII
            self.clear_individual_cache()

            def estimate_execution_time(convergence_factor, execution_time):
                if convergence_factor < 1:
                    return math.log(self.epsilon) / math.log(convergence_factor) * execution_time
                else:
                    return convergence_factor * math.sqrt(self.infinity) * execution_time
            pop, log, hof, evaluation_min_level, evaluation_max_level = \
                optimization_method(pset, initial_population_size, generations, generalization_interval, mu_, lambda_,
                                    crossover_probability, mutation_probability,
                                    min_level, max_level, solver_program, storages, best_expression, evaluation_samples, logbooks,
                                    model_based_estimation=model_based_estimation, pde_parameter_values=pde_parameter_values,
                                    checkpoint_frequency=2, checkpoint=tmp, use_random_search=use_random_search)
            if len(pop[0].fitness.values) == 2:
                pop = sorted(pop, key=lambda ind: estimate_execution_time(ind.fitness.values[0], ind.fitness.values[1]))
                hof = sorted(hof, key=lambda ind: estimate_execution_time(ind.fitness.values[0], ind.fitness.values[1]))
            else:
                pop = sorted(pop, key=lambda ind: ind.fitness.values[0])
                hof = sorted(hof, key=lambda ind: ind.fitness.values[0])
            pops.append(pop)
            hofs.append(hof)
            best_individual = hof[0]
            if self.is_root():
                for individual in hof:
                    if len(individual.fitness.values) == 2:
                        print(f'\nExecution time until convergence: '
                              f'{estimate_execution_time(individual.fitness.values[0], individual.fitness.values[1])}, '
                              # f'Number of iterations: {individual.fitness.values[0]}', flush=True)
                              f'Convergence Factor: {individual.fitness.values[0]}', flush=True)
                    else:
                        print(f'\nExecution time until convergence: {individual.fitness.values[0]}', flush=True)
                    print('Tree representation:', flush=True)
                    print(str(individual), flush=True)

            expression, _ = self.compile_individual(best_individual, pset)
            cycle_function = self.program_generator.generate_cycle_function(expression, storages, min_level, max_level,
                                                                            self.max_level)
            if self.is_root() and min_level == self.min_level:
                average_runtime, average_convergence_factor, average_number_of_iterations = \
                    self.program_generator.generate_and_evaluate(expression, storages, min_level, max_level, solver_program)
                print(f'\nMeasurements for best individual - solving time: {average_runtime}, convergence factor: {average_convergence_factor}, '
                      f'number of iterations: {average_number_of_iterations}')
            solver_program += cycle_function
            self.barrier()

        self.barrier()
        return str(best_individual), pops, logbooks, hofs

    def generate_and_evaluate_program_from_grammar_representation(self, grammar_string: str, maximum_block_size):
        solver_program = ''

        approximation = self.approximation
        rhs = self.rhs
        storages = self._program_generator.generate_storage(self.min_level, self.max_level, self.finest_grid)
        levels = self.max_level - self.min_level
        pset, terminal_list = \
            multigrid_initialization.generate_primitive_set(approximation, rhs, self.dimension,
                                                            self.coarsening_factors, self.max_level, self.equations,
                                                            self.operators, self.fields,
                                                            maximum_local_system_size=maximum_block_size,
                                                            depth=levels,
                                                            LevelFinishedType=self._FinishedType,
                                                            LevelNotFinishedType=self._NotFinishedType)
        self.program_generator.initialize_code_generation(self.min_level, self.max_level, iteration_limit=10000)
        expression, _ = eval(grammar_string, pset.context, {})
        # initial_weights = [1 for _ in relaxation_factor_optimization.obtain_relaxation_factors(expression)]
        # relaxation_factor_optimization.set_relaxation_factors(expression, initial_weights)
        time_to_solution, convergence_factor, number_of_iterations = \
            self._program_generator.generate_and_evaluate(expression, storages, self.min_level, self.max_level,
                                                          solver_program, infinity=self.infinity,
                                                          evaluation_samples=20)

        # print(f'Time: {time_to_solution}, '
        #       f'Convergence factor: {convergence_factor}, '
        #       f'Number of Iterations: {number_of_iterations}', flush=True)
        return time_to_solution, convergence_factor, number_of_iterations

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
    def dump_data_structure(data_structure, file_name):
        import pickle
        with open(file_name, 'wb') as file:
            pickle.dump(data_structure, file)

    @staticmethod
    def load_data_structure(file_name):
        import pickle
        with open(file_name, 'rb') as file:
            return pickle.load(file)
