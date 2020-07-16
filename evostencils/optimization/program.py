import deap.base
from deap import gp, creator, tools
import random
import pickle
import os.path
from evostencils.initialization import multigrid as multigrid_initialization
from evostencils.expressions import base, transformations, system, reference_cycles
from evostencils.genetic_programming import genGrow, mutNodeReplacement, mutInsert, select_unique_best
import evostencils.optimization.relaxation_factors as relaxation_factor_optimization
from evostencils.types import level_control
import math, numpy
import numpy as np
import time
import itertools
import os

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
                 epsilon=1e-12, infinity=1e300, checkpoint_directory_path='./'):
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
        self._weight_optimizer = relaxation_factor_optimization.Optimizer(self)
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

    def reinitialize_code_generation(self, min_level, max_level, program, evaluation_function, number_of_samples=20,
                                     parameter_values={}):
        self.program_generator.reinitialize(min_level, max_level)
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
        self._individual_cache.clear()
        maximum_block_size = 8
        levels_per_run = max_level - min_level
        pset, _ = \
            multigrid_initialization.generate_primitive_set(approximation, rhs, dimension,
                                                            coarsening_factor, max_level, equations,
                                                            operators, fields,
                                                            maximum_block_size=maximum_block_size,
                                                            depth=levels_per_run,
                                                            LevelFinishedType=self._FinishedType,
                                                            LevelNotFinishedType=self._NotFinishedType)
        self.program_generator.initialize_code_generation(min_level, max_level)
        storages = self._program_generator.generate_storage(min_level, max_level, finest_grid)
        self.toolbox.register('evaluate', evaluation_function, pset=pset,
                              storages=storages, min_level=min_level, max_level=max_level,
                              solver_program=program, number_of_samples=number_of_samples,
                              parameter_values=parameter_values)

    @staticmethod
    def _init_creator():
        creator.create("MultiObjectiveFitness", deap.base.Fitness, weights=(-1.0, -1.0))
        creator.create("MultiObjectiveIndividual", gp.PrimitiveTree, fitness=creator.MultiObjectiveFitness)
        creator.create("SingleObjectiveFitness", deap.base.Fitness, weights=(-1.0,))
        creator.create("SingleObjectiveIndividual", gp.PrimitiveTree, fitness=creator.SingleObjectiveFitness)

    def _init_toolbox(self, pset):
        self._toolbox = deap.base.Toolbox()
        self._toolbox.register("expression", genGrow, pset=pset, min_height=0, max_height=50)
        self._toolbox.register("mate", gp.cxOnePoint)

        def mutate(individual, pset):
            operator_choice = random.random()
            if operator_choice < 2.0/3.0:
                return mutInsert(individual, 0, 10, pset)
            else:
                return mutNodeReplacement(individual, pset)

        self._toolbox.register("mutate", mutate, pset=pset)

    def _init_multi_objective_toolbox(self, pset):
        self._toolbox.register("individual", tools.initIterate, creator.MultiObjectiveIndividual,
                               self._toolbox.expression)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)

    def _init_single_objective_toolbox(self, pset):
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

    def mpi_get_left_neighbor(self):
        if self.mpi_rank > 0:
            return self.mpi_rank - 1
        else:
            return self.number_of_mpi_processes - 1

    def mpi_get_right_neighbor(self):
        if self.mpi_rank < self.number_of_mpi_processes - 1:
            return self.mpi_rank + 1
        else:
            return 0

    def mpi_receive_from_neighbors(self):
        right_neighbor = self.mpi_get_right_neighbor()
        left_neighbor = self.mpi_get_left_neighbor()
        left_request = self.mpi_comm.irecv(source=left_neighbor, tag=left_neighbor)
        right_request = self.mpi_comm.irecv(source=right_neighbor, tag=right_neighbor)
        return left_request, right_request

    def mpi_send_to_neighbors(self, data):
        right_neighbor = self.mpi_get_right_neighbor()
        left_neighbor = self.mpi_get_left_neighbor()
        left_request = self.mpi_comm.isend(data, left_neighbor, tag=self.mpi_rank)
        right_request = self.mpi_comm.isend(data, right_neighbor, tag=self.mpi_rank)
        return left_request, right_request

    def mpi_wait_for_receive_request(self, request):
        counter = 0
        cancel_request = False
        try:
            finished = request.Test()
            while not finished:
                if counter == self._timeout_counter_limit:
                    request.Cancel()
                    cancel_request = True
                    break
                counter += 1
                time.sleep(1e-2)
                finished = request.Test()
            if cancel_request:
                print("Communication timeout reached")
                print(f"Immigration of individuals failed on process with rank {self.mpi_rank}")
                return None
            else:
                return request.wait()
        except Exception as e:
            request.Cancel()
            print(e)
            print(f"Immigration of individuals failed on process with rank {self.mpi_rank}")
            return None

    def mpi_wait_for_send_request(self, request):
        counter = 0
        cancel_request = False
        try:
            finished = request.Test()
            while not finished:
                if counter == self._timeout_counter_limit:
                    request.Cancel()
                    cancel_request = True
                    print("Communication timeout reached")
                    print(f"Immigration of individuals failed on process with rank {self.mpi_rank}")
                    break
                counter += 1
                time.sleep(1e-2)
                finished = request.Test()
            if cancel_request:
                return False
            else:
                return True
        except Exception as e:
            request.Cancel()
            print(e)
            print(f"Emigration of individuals failed on process with rank {self.mpi_rank}")
            return False

    def reset_evaluation_counters(self):
        self._failed_evaluations = 0
        self._total_number_of_evaluations = 0

    def generate_individual(self):
        return self._toolbox.individual()

    def compile_individual(self, individual, pset):
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
                or math.isinf(spectral_radius) or numpy.isinf(spectral_radius) or numpy.isnan(spectral_radius):
            values = self.infinity,
            self.add_individual_to_cache(individual, values)
            return values
        else:
            if spectral_radius < 1:
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
                or math.isinf(spectral_radius) or numpy.isinf(spectral_radius) or numpy.isnan(spectral_radius):
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
                                  number_of_samples=20, parameter_values={}):
        self._total_number_of_evaluations += 1
        if self.individual_in_cache(individual):
            return self.get_cached_fitness(individual)
        with suppress_output():
        # with do_nothing():
            try:
                expression1, expression2 = self.compile_individual(individual, pset)
            except MemoryError:
                print("Memory Error", flush=True)
                self._failed_evaluations += 1
                values = self.infinity,
                self.add_individual_to_cache(individual, values)
                return values
            expression = expression1
            initial_weights = relaxation_factor_optimization.obtain_relaxation_factors(expression)
            relaxation_factor_optimization.set_relaxation_factors(expression, initial_weights)
            relaxation_factor_optimization.reset_status(expression)
            n = len(initial_weights)
            tmp = solver_program + self.program_generator.generate_global_weights(n)
            cycle_function = self.program_generator.generate_cycle_function(expression, storages, min_level, max_level,
                                                                            max_level, use_global_weights=True)
            self.program_generator.generate_l3_file(min_level, max_level, tmp + cycle_function, global_values=parameter_values)
            program_generator = self.program_generator
            output_path = program_generator._output_path_generated
            average_time_to_convergence = 0
            average_convergence_factor = 0
            average_number_of_iterations = 0
            failed_testcases = 0
            count = 0
            threshold = number_of_samples // 4

            program_generator.run_exastencils_compiler(knowledge_path=program_generator.knowledge_path_generated,
                                                       settings_path=program_generator.settings_path_generated)
            weights = []
            for _ in range(n):
                w = random.gauss(1.0, 0.2)
                while w < 0 or w > 2.0:
                    w = random.gauss(1.0, 0.2)
                weights.append(w)
            for i in range(number_of_samples):
                if failed_testcases > threshold:
                    break
                program_generator.generate_global_weight_initializations(output_path, weights)
                program_generator.run_c_compiler(output_path)
                time_to_convergence, convergence_factor, number_of_iterations = \
                    program_generator.evaluate(output_path,
                                               infinity=self.infinity,
                                               number_of_samples=1)
                if number_of_iterations >= self.infinity or convergence_factor > 1:
                    failed_testcases += 1
                average_time_to_convergence += time_to_convergence / number_of_samples
                average_convergence_factor += convergence_factor / number_of_samples
                average_number_of_iterations += number_of_iterations / number_of_samples
                program_generator.restore_global_initializations(output_path)
                count += 1
            # time_to_convergence, convergence_factor, number_of_iterations = \
            #         self._program_generator.generate_and_evaluate(expression, storages, min_level, max_level,
            #                 solver_program, infinity=self.infinity,
            #                 number_of_samples=3)
            fitness = average_time_to_convergence,
            if average_number_of_iterations >= self.infinity / number_of_samples:
                if failed_testcases >= threshold:
                    average_convergence_factor = average_convergence_factor * number_of_samples / count
                    average_number_of_iterations = average_number_of_iterations * number_of_samples / count
                return average_convergence_factor**0.5 * average_number_of_iterations**0.5,
            self.add_individual_to_cache(individual, fitness)
            return fitness

    def evaluate_multiple_objectives(self, individual, pset, storages, min_level, max_level, solver_program,
                                     number_of_samples=20, parameter_values={}):
        self._total_number_of_evaluations += 1
        if self.individual_in_cache(individual):
            return self.get_cached_fitness(individual)
        with suppress_output():
            try:
                expression1, expression2 = self.compile_individual(individual, pset)
            except MemoryError:
                self._failed_evaluations += 1
                values = self.infinity, self.infinity
                self.add_individual_to_cache(individual, values)
                return values
            expression = expression1
            initial_weights = relaxation_factor_optimization.obtain_relaxation_factors(expression)
            relaxation_factor_optimization.set_relaxation_factors(expression, initial_weights)
            relaxation_factor_optimization.reset_status(expression)
            n = len(initial_weights)
            tmp = solver_program + self.program_generator.generate_global_weights(n)
            cycle_function = self.program_generator.generate_cycle_function(expression, storages, min_level, max_level,
                                                                            max_level, use_global_weights=True)
            self.program_generator.generate_l3_file(min_level, max_level, tmp + cycle_function, global_values=parameter_values)
            program_generator = self.program_generator
            output_path = program_generator._output_path_generated
            average_time_to_convergence = 0
            average_convergence_factor = 0
            average_number_of_iterations = 0
            failed_testcases = 0
            threshold = number_of_samples // 4

            program_generator.run_exastencils_compiler(knowledge_path=program_generator.knowledge_path_generated,
                                                       settings_path=program_generator.settings_path_generated)
            weights = []
            for _ in range(n):
                w = random.gauss(1.0, 0.2)
                while w < 0 or w > 2.0:
                    w = random.gauss(1.0, 0.2)
                weights.append(w)
            count = 0
            for i in range(number_of_samples):
                if failed_testcases > threshold:
                    break
                program_generator.generate_global_weight_initializations(output_path, weights)
                program_generator.run_c_compiler(output_path)
                time_to_convergence, convergence_factor, number_of_iterations = \
                    program_generator.evaluate(output_path,
                                               infinity=self.infinity,
                                               number_of_samples=1)
                if number_of_iterations >= self.infinity or convergence_factor > 1:
                    failed_testcases += 1
                    average_time_to_convergence += self.infinity / number_of_samples
                    average_number_of_iterations += self.infinity / number_of_samples
                else:
                    average_time_to_convergence += time_to_convergence / number_of_samples
                    average_number_of_iterations += number_of_iterations / number_of_samples
                average_convergence_factor += convergence_factor / number_of_samples
                program_generator.restore_global_initializations(output_path)
                count += 1
            values = average_number_of_iterations, average_time_to_convergence / average_number_of_iterations
            if average_number_of_iterations >= self.infinity / number_of_samples:
                if failed_testcases >= threshold:
                    average_convergence_factor = average_convergence_factor * number_of_samples / count
                    average_number_of_iterations = average_number_of_iterations * number_of_samples / count
                return average_convergence_factor**0.5 * average_number_of_iterations**0.5, self.infinity
            self.add_individual_to_cache(individual, values)
            return values

    def multi_objective_random_search(self, pset, initial_population_size, generations, mu_, lambda_,
                                      _, __, min_level, max_level,
                                      program, storages, solver, logbooks, checkpoint_frequency=2, checkpoint=None):

        if self.is_root():
            print("Running Multi-Objective Random Search Genetic Programming", flush=True)
        self._init_multi_objective_toolbox(pset)
        self._toolbox.register("select", tools.selNSGA2, nd='standard')
        # self._toolbox.register('evaluate', self.estimate_multiple_objectives, pset=pset)
        self._toolbox.register('evaluate', self.evaluate_multiple_objectives, pset=pset,
                               storages=storages, min_level=min_level, max_level=max_level,
                               solver_program=program)

        stats_fit1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fit2 = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_size = tools.Statistics(len)

        mstats = tools.MultiStatistics(convergence_factor=stats_fit1, runtime=stats_fit2, size=stats_size)

        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        hof = tools.ParetoFront(similar=lambda a, b: a.fitness == b.fitness)

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
            population = self.toolbox.population(n=initial_population_size)
            min_generation = 0
        max_generation = generations

        if use_checkpoint:
            logbook = logbooks[-1]
        else:
            logbook = tools.Logbook()
            logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
            logbooks.append(logbook)

        invalid_ind = [ind for ind in population]
        self.reset_evaluation_counters()
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        hof.update(population)
        population = self.toolbox.select(population, len(population))
        record = mstats.compile(population) if mstats is not None else {}
        logbook.record(gen=min_generation, nevals=len(invalid_ind), **record)

        if self.is_root():
            print(logbook.stream, flush=True)
        # Begin the generational process
        for gen in range(min_generation + 1, max_generation + 1):
            # Vary the population
            offspring = self._toolbox.population(n=lambda_)
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
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
                    print(f'Skipping checkpoint on process with rank {self.mpi_rank}', flush=True)
            # Select the next generation population
            population[:] = self.toolbox.select(population + offspring, mu_)
            record = mstats.compile(population)
            # Update the statistics with the new population
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if self.is_root():
                print(logbook.stream, flush=True)

        return population, logbook, hof

    @staticmethod
    def compute_average_population_execution_time(population):
        avg_execution_time = 0
        for ind in population:
            execution_time = 1
            for value in ind.fitness.values:
                execution_time *= value
            avg_execution_time += execution_time / len(population) / 1000
        return avg_execution_time

    def ea_mu_plus_lambda(self, initial_population_size, generations, mu_, lambda_,
                          crossover_probability, mutation_probability, min_level, max_level,
                          program, solver, logbooks, parameter_values, checkpoint_frequency, checkpoint, mstats, hof):

        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

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

        invalid_ind = [ind for ind in population]
        self.reset_evaluation_counters()
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        successful_evaluations = self._total_number_of_evaluations - self._failed_evaluations
        if self.is_root():
            print("Number of successful evaluations in initial population:",
                  successful_evaluations, flush=True)
        self.reset_evaluation_counters()
        population = self.toolbox.select(population, max(lambda_, successful_evaluations - successful_evaluations % 4))
        hof.update(population)
        record = mstats.compile(population) if mstats is not None else {}
        logbook.record(gen=min_generation, nevals=len(invalid_ind), **record)
        if self.is_root():
            print(logbook.stream, flush=True)
        # Begin the generational process
        immigration_interval = 5
        receive_request_left_neighbor = None
        receive_request_right_neighbor = None
        if self.number_of_mpi_processes > 1:
            receive_request_left_neighbor, receive_request_right_neighbor = self.mpi_receive_from_neighbors()
        execution_time_threshold = 1.5
        count = 0
        evaluation_min_level = min_level
        evaluation_max_level = max_level
        level_offset = 0
        for gen in range(min_generation + 1, max_generation + 1):
            average_execution_time = self.compute_average_population_execution_time(population)
            print("Average execution time:", average_execution_time, flush=True)
            if count >= 10 and average_execution_time < execution_time_threshold:
                level_offset += 1
                evaluation_min_level = min_level + level_offset
                evaluation_max_level = max_level + level_offset
                next_parameter_values = {}
                for key, values in parameter_values.items():
                    assert level_offset < len(values), 'Too few parameter values provided'
                    next_parameter_values[key] = values[level_offset]
                count = 0
                print("Increasing problem size", flush=True)
                self.reinitialize_code_generation(evaluation_min_level, evaluation_max_level, program,
                                                  self.evaluate_multiple_objectives, parameter_values=next_parameter_values)
                invalid_ind = [ind for ind in population]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            if gen % immigration_interval == 0 and self.number_of_mpi_processes > 1:
                if self.is_root():
                    print("Exchanging colonies", flush=True)

                send_request_left_neighbor, send_request_right_neighbor = \
                    self.mpi_send_to_neighbors(population[:len(population)//2])

                self.mpi_wait_for_send_request(send_request_left_neighbor)
                left_neighbor_population = self.mpi_wait_for_receive_request(receive_request_left_neighbor)
                if left_neighbor_population is not None:
                    population.extend(left_neighbor_population)

                self.mpi_wait_for_send_request(send_request_right_neighbor)
                right_neighbor_population = self.mpi_wait_for_receive_request(receive_request_right_neighbor)
                if right_neighbor_population is not None:
                    population.extend(right_neighbor_population)

                receive_request_left_neighbor, receive_request_right_neighbor = self.mpi_receive_from_neighbors()
                population = self.toolbox.select(population, mu_)
                self.mpi_comm.barrier()
            # Vary the population
            selected = self.toolbox.select_for_mating(population, lambda_)
            parents = [self.toolbox.clone(ind) for ind in selected]
            offspring = []
            for ind1, ind2 in zip(parents[::2], parents[1::2]):
                operator_choice = random.random()
                if operator_choice < crossover_probability:
                    child1, child2 = self.toolbox.mate(ind1, ind2)
                elif operator_choice < crossover_probability + mutation_probability + self.epsilon:
                    child1, = self.toolbox.mutate(ind1)
                    child2, = self.toolbox.mutate(ind2)
                else:
                    child1 = ind1
                    child2 = ind2
                del child1.fitness.values, child2.fitness.values
                offspring.append(child1)
                offspring.append(child2)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            hof.update(offspring)

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
            population = self.toolbox.select(population + offspring, mu_)
            count += 1
            record = mstats.compile(population)
            # Update the statistics with the new population
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if self.is_root():
                print(logbook.stream, flush=True)
        if self.is_root():
            print("Exchanging colonies", flush=True)

        if self.number_of_mpi_processes > 1:
            send_request_left_neighbor, send_request_right_neighbor = \
                self.mpi_send_to_neighbors(population[:len(population) // 2])

            self.mpi_wait_for_send_request(send_request_left_neighbor)
            left_neighbor_population = self.mpi_wait_for_receive_request(receive_request_left_neighbor)
            if left_neighbor_population is not None:
                population.extend(left_neighbor_population)

            self.mpi_wait_for_send_request(send_request_right_neighbor)
            right_neighbor_population = self.mpi_wait_for_receive_request(receive_request_right_neighbor)
            if right_neighbor_population is not None:
                population.extend(right_neighbor_population)

            self.mpi_comm.barrier()
            number_of_immigrants = max(10, 2 * len(population) // self.number_of_mpi_processes)
            colonies = self.mpi_comm.allgather(population[:number_of_immigrants])
            immigrants = list(itertools.chain.from_iterable(colonies))
            population.extend(immigrants)

        hof.update(population)
        if self.is_root():
            print("Optimization finished", flush=True)

        return population, logbook, hof, evaluation_min_level, evaluation_max_level

    def SOGP(self, pset, initial_population_size, generations, mu_, lambda_,
             crossover_probability, mutation_probability, min_level, max_level,
             program, storages, solver, logbooks, parameter_values={}, checkpoint_frequency=2, checkpoint=None):
        if self.is_root():
            print("Running Single-Objective Genetic Programming", flush=True)
        self._init_single_objective_toolbox(pset)
        self._toolbox.register("select", select_unique_best)
        self._toolbox.register("select_for_mating", tools.selTournament, tournsize=2)
        # self._toolbox.register('evaluate', self.estimate_single_objective, pset=pset)
        initial_parameter_values = {}
        for key, values in parameter_values.items():
            initial_parameter_values[key] = values[0]
        self._toolbox.register('evaluate', self.evaluate_single_objective, pset=pset,
                               storages=storages, min_level=min_level, max_level=max_level, solver_program=program,
                               parameter_values=initial_parameter_values)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

        hof = tools.HallOfFame(100, similar=lambda a, b: a.fitness == b.fitness)
        return self.ea_mu_plus_lambda(initial_population_size, generations, mu_, lambda_,
                                      crossover_probability, mutation_probability, min_level, max_level,
                                      program, solver, logbooks, parameter_values, checkpoint_frequency, checkpoint, mstats, hof)

    def NSGAII(self, pset, initial_population_size, generations, mu_, lambda_,
               crossover_probability, mutation_probability, min_level, max_level,
               program, storages, solver, logbooks, parameter_values={}, checkpoint_frequency=2, checkpoint=None):
        if self.is_root():
            print("Running NSGA-II Genetic Programming", flush=True)
        self._init_multi_objective_toolbox(pset)
        self._toolbox.register("select", tools.selNSGA2, nd='standard')
        self._toolbox.register("select_for_mating", tools.selTournamentDCD)
        # self._toolbox.register('evaluate', self.estimate_multiple_objectives, pset=pset)
        initial_parameter_values = {}
        for key, values in parameter_values.items():
            initial_parameter_values[key] = values[0]
        self._toolbox.register('evaluate', self.evaluate_multiple_objectives, pset=pset,
                               storages=storages, min_level=min_level, max_level=max_level,
                               solver_program=program, parameter_values=initial_parameter_values)

        stats_fit1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fit2 = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(number_of_iterations=stats_fit1, execution_time=stats_fit2, size=stats_size)

        hof = tools.ParetoFront(similar=lambda a, b: a.fitness == b.fitness)

        return self.ea_mu_plus_lambda(initial_population_size, generations, mu_, lambda_,
                                      crossover_probability, mutation_probability, min_level, max_level,
                                      program, solver, logbooks, parameter_values, checkpoint_frequency, checkpoint, mstats, hof)

    def NSGAIII(self, pset, initial_population_size, generations, mu_, lambda_,
                crossover_probability, mutation_probability, min_level, max_level,
                program, storages, solver, logbooks, parameter_values={}, checkpoint_frequency=2, checkpoint=None):
        if self.is_root():
            print("Running NSGA-III Genetic Programming", flush=True)
        self._init_multi_objective_toolbox(pset)
        H = mu_
        reference_points = tools.uniform_reference_points(2, H)
        mu_ = H + (4 - H % 4)
        self._toolbox.register("select", tools.selNSGA3WithMemory(reference_points, nd='standard'))
        self._toolbox.register("select_for_mating", tools.selRandom)
        # self._toolbox.register('evaluate', self.estimate_multiple_objectives, pset=pset)
        initial_parameter_values = {}
        for key, values in parameter_values.items():
            initial_parameter_values[key] = values[0]
        self._toolbox.register('evaluate', self.evaluate_multiple_objectives, pset=pset,
                               storages=storages, min_level=min_level, max_level=max_level,
                               solver_program=program, parameter_values=initial_parameter_values)

        stats_fit1 = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_fit2 = tools.Statistics(lambda ind: ind.fitness.values[1])
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(number_of_iterations=stats_fit1, execution_time=stats_fit2, size=stats_size)

        hof = tools.ParetoFront(similar=lambda a, b: a.fitness == b.fitness)

        return self.ea_mu_plus_lambda(initial_population_size, generations, mu_, lambda_,
                                      crossover_probability, mutation_probability, min_level, max_level,
                                      program, solver, logbooks, parameter_values, checkpoint_frequency, checkpoint, mstats, hof)

    def evolutionary_optimization(self, levels_per_run=2, gp_mu=100, gp_lambda=100, gp_generations=100,
                                  gp_crossover_probability=0.5, gp_mutation_probability=0.5, es_generations=200,
                                  required_convergence=0.9,
                                  restart_from_checkpoint=False, maximum_block_size=8, optimization_method=None,
                                  optimize_relaxation_factors=True,
                                  parameter_values={},
                                  krylov_subspace_methods=('ConjugateGradient', 'BiCGStab', 'MinRes',
                                                           'ConjugateResidual'),
                                  minimum_solver_iterations=8, maximum_solver_iterations=1024):
        assert minimum_solver_iterations < maximum_solver_iterations, 'Invalid range of solver iterations'

        levels = self.max_level - self.min_level
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
        residual_norm_functions = \
            self.program_generator.generate_cached_krylov_subspace_solvers(self.min_level + 1, self.max_level,
                                                                           krylov_subspace_methods,
                                                                           minimum_solver_iterations,
                                                                           maximum_solver_iterations)
        for residual_norm_function in residual_norm_functions[:len(residual_norm_functions) - 1]:
            solver_program += residual_norm_function
            solver_program += '\n'
        for i in range(0, levels, levels_per_run):
            min_level = self.max_level - (i + levels_per_run)
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
            if i > 0:
                self.performance_evaluator.set_runtime_of_coarse_grid_solver(0.0)

            rhs = right_hand_sides[i]
            pset, terminal_list = \
                multigrid_initialization.generate_primitive_set(approximation, rhs, self.dimension,
                                                                self.coarsening_factors, max_level, self.equations,
                                                                self.operators, self.fields,
                                                                maximum_block_size=maximum_block_size,
                                                                depth=levels_per_run,
                                                                LevelFinishedType=self._FinishedType,
                                                                LevelNotFinishedType=self._NotFinishedType,
                                                                krylov_subspace_methods=krylov_subspace_methods,
                                                                minimum_solver_iterations=minimum_solver_iterations,
                                                                maximum_solver_iterations=maximum_solver_iterations)
            self._init_toolbox(pset)
            tmp = None
            if pass_checkpoint:
                tmp = checkpoint
            initial_population_size = 4 * gp_mu
            initial_population_size -= initial_population_size % 4
            gp_mu -= gp_mu % 4
            gp_lambda -= gp_lambda % 4

            self.program_generator._counter = 0
            self.program_generator._average_generation_time = 0

            self.program_generator.initialize_code_generation(self.min_level, self.max_level, iteration_limit=128)
            if optimization_method is None:
                optimization_method = self.NSGAII
            self.clear_individual_cache()
            pop, log, hof, evaluation_min_level, evaluation_max_level = \
                optimization_method(pset, initial_population_size, gp_generations, gp_mu, gp_lambda,
                                    gp_crossover_probability, gp_mutation_probability,
                                    min_level, max_level, solver_program, storages, best_expression, logbooks, parameter_values=parameter_values,
                                    checkpoint_frequency=2, checkpoint=tmp)

            pops.append(pop)
            best_time = self.infinity
            best_number_of_iterations = self.infinity
            evaluation_min_level += 1
            evaluation_max_level += 1
            self.reinitialize_code_generation(evaluation_min_level, evaluation_max_level, solver_program,
                                              self.evaluate_multiple_objectives, number_of_samples=50)
            output_directory_path = f'./hall_of_fame_{i}_{self.program_generator.problem_name}'
            if not os.path.exists(output_directory_path):
                os.makedirs(output_directory_path)
            hof = sorted(hof, key=lambda ind: ind.fitness.values[0])
            try:
                for j in range(0, min(len(hof), 100)):
                    individual = hof[j]
                    if individual.fitness.values[0] >= self.infinity:
                        break
                    values = self.toolbox.evaluate(individual)
                    individual.fitness.values = values
                    time = individual.fitness.values[0] * individual.fitness.values[1]
                    number_of_iterations = individual.fitness.values[0]
                    if self.is_root():
                        print(f'\nExecution time until convergence: {time}, '
                              f'Number of Iterations: {number_of_iterations}', flush=True)
                        with open(f'{output_directory_path}/individual_{j}.txt', 'w') as grammar_file:
                            grammar_file.write(str(individual) + '\n')
                        print('Tree representation:', flush=True)
                        print(str(individual), flush=True)

                    if time < best_time: #  and convergence_factor < required_convergence:
                        best_individual = individual
                        best_time = time
                        best_number_of_iterations = number_of_iterations

            except (KeyboardInterrupt, Exception) as e:
                raise e
            self.mpi_comm.barrier()
            print(f"Rank {self.mpi_rank} - Fastest execution time until convergence: {best_time}, "
                  f"Number of iterations: {best_number_of_iterations}",
                  flush=True)
            #TODO fix relaxation factor optimization
            """
            if optimize_relaxation_factors and es_generations > 0:
                relaxation_factors, improved_convergence_factor = \
                    self.optimize_relaxation_factors(best_expression, es_generations, evaluation_min_level, evaluation_max_level,
                                                 solver_program, storages, best_time)
                relaxation_factor_optimization.set_relaxation_factors(best_expression, relaxation_factors)
                self.program_generator.initialize_code_generation(self.min_level, self.max_level)
                best_time, convergence_factor, number_of_iterations = \
                    self._program_generator.generate_and_evaluate(best_expression, storages, evaluation_min_level, evaluation_max_level,
                                                              solver_program, infinity=self.infinity,
                                                              number_of_samples=10)
                best_expression.runtime = best_time / number_of_iterations * 1e-3
                self.mpi_comm.barrier()
                print(f"Rank {self.mpi_rank} - Improved time: {best_time}, Number of Iterations: {number_of_iterations}",
                      flush=True)
            """
        self.mpi_comm.barrier()
        return str(best_individual), pops, logbooks

    def optimize_relaxation_factors(self, expression, generations, min_level, max_level, base_program, storages, evaluation_time):
        initial_weights = relaxation_factor_optimization.obtain_relaxation_factors(expression)
        relaxation_factor_optimization.set_relaxation_factors(expression, initial_weights)
        relaxation_factor_optimization.reset_status(expression)
        n = len(initial_weights)
        self.program_generator.initialize_code_generation(self.min_level, self.max_level, iteration_limit=64)
        try:
            tmp = base_program + self.program_generator.generate_global_weights(n)
            cycle_function = self.program_generator.generate_cycle_function(expression, storages, min_level, max_level,
                                                                            self.max_level, use_global_weights=True)
            self.program_generator.generate_l3_file(min_level, self.max_level, tmp + cycle_function)
            best_individual = self._weight_optimizer.optimize(expression, n, generations, storages, evaluation_time)
            best_weights = list(best_individual)
            time_to_solution, = best_individual.fitness.values
        except (KeyboardInterrupt, Exception) as e:
            raise e
        return best_weights, time_to_solution

    def generate_and_evaluate_program_from_grammar_representation(self, grammar_string: str, maximum_block_size,
                                                                  optimize_relaxation_factors=True,
                                                                  krylov_subspace_methods=('ConjugateGradient',
                                                                                           'BiCGStab', 'MinRes',
                                                                                           'ConjugateResidual'),
                                                                  minimum_solver_iterations=8,
                                                                  maximum_solver_iterations=1024):
        assert minimum_solver_iterations < maximum_solver_iterations, 'Invalid range of solver iterations'
        solver_program = ''

        approximation = self.approximation
        rhs = self.rhs
        storages = self._program_generator.generate_storage(self.min_level, self.max_level, self.finest_grid)
        solver_list = krylov_subspace_methods
        iteration_list = (2**i for i in range(int(math.log2(minimum_solver_iterations)),
                                              int(math.log2(maximum_solver_iterations))))
        tmp = minimum_solver_iterations
        minimum_solver_iterations = maximum_solver_iterations
        maximum_solver_iterations = tmp

        for i in iteration_list:
            if str(i) in grammar_string:
                if i < minimum_solver_iterations:
                    minimum_solver_iterations = i
                if i > maximum_solver_iterations:
                    maximum_solver_iterations = i

        residual_norm_functions = \
            self.program_generator.generate_cached_krylov_subspace_solvers(self.min_level + 1, self.max_level,
                                                                           solver_list,
                                                                           minimum_solver_iterations,
                                                                           maximum_solver_iterations)
        for residual_norm_function in residual_norm_functions[:len(residual_norm_functions) - 1]:
            solver_program += residual_norm_function
            solver_program += '\n'
        levels = self.max_level - self.min_level
        pset, terminal_list = \
            multigrid_initialization.generate_primitive_set(approximation, rhs, self.dimension,
                                                            self.coarsening_factors, self.max_level, self.equations,
                                                            self.operators, self.fields,
                                                            maximum_block_size=maximum_block_size,
                                                            depth=levels,
                                                            LevelFinishedType=self._FinishedType,
                                                            LevelNotFinishedType=self._NotFinishedType,
                                                            krylov_subspace_methods=krylov_subspace_methods,
                                                            minimum_solver_iterations=minimum_solver_iterations,
                                                            maximum_solver_iterations=maximum_solver_iterations)
        self.program_generator.initialize_code_generation(self.min_level, self.max_level, iteration_limit=128)
        expression, _ = eval(grammar_string, pset.context, {})
        # initial_weights = [1 for _ in relaxation_factor_optimization.obtain_relaxation_factors(expression)]
        # relaxation_factor_optimization.set_relaxation_factors(expression, initial_weights)
        time_to_solution, convergence_factor, number_of_iterations = \
            self._program_generator.generate_and_evaluate(expression, storages, self.min_level, self.max_level,
                                                          solver_program, infinity=self.infinity,
                                                          number_of_samples=20)

        print(f'Time: {time_to_solution}, '
              f'Convergence factor: {convergence_factor}, '
              f'Number of Iterations: {number_of_iterations}', flush=True)
        if optimize_relaxation_factors:
            relaxation_factors, _ = \
                self.optimize_relaxation_factors(expression, 150, self.min_level, self.max_level, solver_program,
                                                 storages, time_to_solution)
            relaxation_factor_optimization.set_relaxation_factors(expression, relaxation_factors)
            self.program_generator.initialize_code_generation(self.min_level, self.max_level, iteration_limit=128)
            time_to_solution, convergence_factor, number_of_iterations = \
                self._program_generator.generate_and_evaluate(expression, storages, self.min_level, self.max_level,
                                                              solver_program, infinity=self.infinity,
                                                              number_of_samples=20)
            print(f'Time: {time_to_solution}, '
                  f'Convergence factor: {convergence_factor}, '
                  f'Number of Iterations: {number_of_iterations}', flush=True)

        cycle_function = self.program_generator.generate_cycle_function(expression, storages, self.min_level,
                                                                        self.max_level, self.max_level)
        return cycle_function

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
