import numpy as np
import deap.base
from deap import gp, creator, tools
import random
import pickle
import os.path
from evostencils.initialization import multigrid as multigrid_initialization
from evostencils.expressions import base, transformations, system
from evostencils.deap_extension import genGrow, AST
import evostencils.optimization.weights as weights
from evostencils.types import level_control
import math


# Define a context manager to suppress stdout and stderr.
class suppress_stdout_stderr(object):
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
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
                 program_generator=None, epsilon=1e-20, infinity=1e100, checkpoint_directory_path='./'):
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
        self._weight_optimizer = weights.Optimizer(self)

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

    def evaluate(self, individual, pset):
        import numpy, math
        if len(individual) > 150:
            return self.infinity, self.infinity
        try:
            expression1, expression2 = self.compile_individual(individual, pset)
        except MemoryError:
            return self.infinity, self.infinity

        expression = expression1
        with suppress_stdout_stderr():
            spectral_radius = self.convergence_evaluator.compute_spectral_radius(expression)

        if spectral_radius == 0.0 or math.isnan(spectral_radius) \
                or math.isinf(spectral_radius) or numpy.isinf(spectral_radius) or numpy.isnan(spectral_radius):
            return self.infinity, self.infinity
        else:
            if self._performance_evaluator is not None:
                runtime = self.performance_evaluator.estimate_runtime(expression) * 1e3 # ms
                return spectral_radius, runtime
            else:
                return spectral_radius, self.infinity

    def nsgaII(self, initial_population_size, generations, mu_, lambda_, crossover_probability, mutation_probability,
               min_level, max_level, program, solver, logbooks, checkpoint_frequency=5, checkpoint=None):
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
                # TODO fix problem with invalidate expression
                if solver is not None:
                    transformations.invalidate_expression(solver)
                logbooks[-1] = logbook
                checkpoint = CheckPoint(min_level, max_level, gen, program, solver, pop, logbooks)
                # checkpoint.dump_to_file(f'{self._checkpoint_directory_path}/checkpoint.p')
            # Select the next generation population
            pop = toolbox.select(pop + offspring, mu_)
            record = mstats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            print(logbook.stream)

        return pop, logbook, hof

    def default_optimization(self, gp_mu=20, gp_lambda=20, gp_generations=20, gp_crossover_probability=0.7,
                             gp_mutation_probability=0.3, es_generations=50, required_convergence=0.9,
                             restart_from_checkpoint=False, maximum_block_size=2):

        levels = self.max_level - self.min_level
        levels_per_run = 2
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
            # Not clear if needed
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

            pop, log, hof = self.nsgaII(10 * gp_mu, gp_generations, gp_mu, gp_lambda, gp_crossover_probability,
                                        gp_mutation_probability, min_level, max_level, solver_program, best_expression, logbooks,
                                        checkpoint_frequency=5, checkpoint=tmp)

            pops.append(pop)

            def estimated_solving_time(ind):
                rho = ind.fitness.values[0]
                if rho < 1.0:
                    return math.log(self.epsilon) / math.log(rho) * ind.fitness.values[1]
                else:
                    return rho * self.infinity
            hof = sorted(hof, key=estimated_solving_time)
            best_time = self.infinity
            best_individual = hof[0]
            self._program_generator._counter = 0
            self._program_generator._average_generation_time = 0
            try:
                self._program_generator.generate_level_adapted_knowledge_file(max_level)
                self._program_generator.run_exastencils_compiler()
                for j in range(0, min(50, len(hof))):
                    if j < len(hof) - 1 and abs(hof[j].fitness.values[0] - hof[j + 1].fitness.values[0]) < self.epsilon and \
                            abs(hof[j].fitness.values[1] - hof[j + 1].fitness.values[1] < self.epsilon):
                        continue
                    individual = hof[j]
                    if individual.fitness.values[0] > 1.0:
                        break
                    expression = self.compile_individual(individual, pset)[0]

                    time, convergence_factor = \
                        self._program_generator.generate_and_evaluate(expression, storages, min_level, max_level,
                                                                      solver_program, number_of_samples=5)
                    print(f'Time: {time}, Estimated convergence factor: {individual.fitness.values[0]}, '
                          f'Measured convergence factor: {convergence_factor}')
                    individual.fitness.values = (convergence_factor, individual.fitness.values[1])
                    if time < best_time and convergence_factor < required_convergence:
                        best_individual = individual
                        best_time = time
            except Exception as e:
                self._program_generator.restore_files()
                raise e
            best_convergence_factor = best_individual.fitness.values[0]
            print(f"Best individual: ({best_convergence_factor}), ({best_individual.fitness.values[1]})")
            best_expression = self.compile_individual(best_individual, pset)[0]
            cycle_function = self._program_generator.generate_cycle_function(best_expression, storages, min_level,
                                                                             max_level, self.max_level)
            solver_program += cycle_function
            print(f"ExaSlang representation:\n{cycle_function}\n")
            best_expression.evaluate = False
            self.convergence_evaluator.compute_spectral_radius(best_expression)
            self.performance_evaluator.estimate_runtime(best_expression)

        return pops, logbooks

    def optimize_weights(self, expression, generations, base_program=None, storages=None):
        # expression = self.compile_expression(individual)
        initial_weights = weights.obtain_weights(expression)
        weights.set_weights(expression, initial_weights)
        weights.reset_status(expression)
        n = len(initial_weights)
        if base_program is not None and storages is not None:
            tmp = base_program + self.program_generator.generate_global_weights(n)
            evaluation_program = tmp + self.program_generator.generate_cycle_function(expression, storages,
                                                                                      use_global_weights=True)
            self.program_generator.write_program_to_file(evaluation_program)
            self.program_generator.run_exastencils_compiler()
        best_individual = self._weight_optimizer.optimize(expression, n, generations, base_program, storages)
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
        ax2.set_ylabel("Runtime (ms)", color="r")
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
        Optimizer.plot_multiobjective_data(gen, convergence_mins, runtime_mins, 'Minimum Spectral Radius', 'Minimum Estimated Runtime per Iteration')

    @staticmethod
    def plot_average_fitness(logbook):
        gen = logbook.select("gen")
        convergence_avgs = logbook.chapters["convergence"].select("avg")
        runtime_avgs = logbook.chapters["runtime"].select("avg")
        Optimizer.plot_multiobjective_data(gen, convergence_avgs, runtime_avgs, 'Average Spectral Radius', 'Average Estimated Runtime per Iteration')

    @staticmethod
    def plot_pareto_front(pop):
        import matplotlib.pyplot as plt
        import numpy
        pop.sort(key=lambda x: x.fitness.values)

        front = numpy.array([ind.fitness.values for ind in pop])
        plt.scatter(front[:, 0], front[:, 1], c="b")
        plt.xlabel("Spectral Radius")
        plt.ylabel("Runtime (ms)")
        plt.axis("tight")
        plt.show()

    @staticmethod
    def dump_population(pop, file_name):
        import pickle
        with open(file_name, 'wb') as file:
            pickle.dump(pop, file)



