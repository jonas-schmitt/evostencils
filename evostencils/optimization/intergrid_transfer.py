from evostencils.ir import base, system
from evostencils.code_generation.exastencils import ProgramGenerator
from deap import creator, tools, algorithms, cma
import deap
from math import log
import numpy
from evostencils.stencils import constant


def optimize(iterations, program_generator: ProgramGenerator, max_level,
             operator: system.Operator, default_restriction: system.Restriction, default_prolongation: system.Prolongation,
             approximation: system.Approximation, coarse_approximation: system.Approximation,
             rhs: system.RightHandSide,
             operator_range=2):
    number_of_prolongation_weights = []
    number_of_restriction_weights = []
    grid = approximation.grid
    coarse_grid = coarse_approximation.grid
    for g in grid:
        dimension = g.dimension
        width = 2 * operator_range + 1
        tmp = 1
        for _ in range(dimension):
            tmp *= width
        number_of_restriction_weights.append(tmp)
        number_of_prolongation_weights.append(tmp)
    problem_size = 0
    for n, m in zip(number_of_restriction_weights, number_of_prolongation_weights):
        problem_size += n + m

    def generate_prolongation_and_restriction(weights_, default_restriction_: system.Restriction, default_prolongation_: system.Prolongation):
        offset = 0
        restriction_operators = []
        prolongation_operators = []
        default_restriction_entries = default_restriction_.entries
        default_prolongation_entries = default_prolongation_.entries
        for kk, nn in enumerate(number_of_restriction_weights):
            restriction_weights = weights_[offset:nn]
            offset += nn
            entries = []
            index = 0
            for ii in range(-operator_range, operator_range + 1):
                for jj in range(-operator_range, operator_range + 1):
                    entries.append(((ii, jj), restriction_weights[index]))
                    index += 1
            stencil = constant.Stencil(entries)
            stencil_generator = base.ConstantStencilGenerator(stencil)
            restriction_ = base.Restriction(default_restriction_entries[kk][kk].name, grid[kk], coarse_grid[kk], stencil_generator)
            restriction_operators.append(restriction_)
        for kk, nn in enumerate(number_of_prolongation_weights):
            prolongation_weights = weights_[offset:(nn + offset)]
            offset += nn
            entries = []
            index = 0
            for ii in range(-operator_range, operator_range + 1):
                for jj in range(-operator_range, operator_range + 1):
                    entries.append(((ii, jj), prolongation_weights[index]))
                    index += 1
            stencil = constant.Stencil(entries)
            stencil_generator = base.ConstantStencilGenerator(stencil)
            prolongation = base.Prolongation(default_prolongation_entries[kk][kk].name, grid[kk], coarse_grid[kk], stencil_generator)
            prolongation_operators.append(prolongation)
        restriction_ = system.Restriction(default_restriction_.name, restriction_operators)
        prolongation = system.Prolongation(default_prolongation_.name, prolongation_operators)
        return restriction_, prolongation

    def generate_coarse_grid_correction(restriction, prolongation, omega=1):
        # residual = base.Residual(operator, approximation, rhs)
        # correction = base.Multiplication(base.Inverse(smoother.generate_collective_jacobi(operator)), residual)
        # cycle = base.Cycle(approximation, rhs, correction, partitioning=part.RedBlack, relaxation_factor=1)
        cycle = approximation
        residual = base.Residual(operator, cycle, rhs)
        coarse_grid_correction = base.Multiplication(restriction, residual)
        coarse_grid_correction = \
            base.Multiplication(base.CoarseGridSolver(system.get_coarse_operator(operator, coarse_approximation.grid)),
                                coarse_grid_correction)
        coarse_grid_correction = base.Multiplication(prolongation, coarse_grid_correction)
        cycle = base.Cycle(cycle, rhs, coarse_grid_correction, relaxation_factor=omega)
        # residual = base.Residual(operator, cycle, rhs)
        # correction = base.Multiplication(base.Inverse(smoother.generate_collective_jacobi(operator)), residual)
        # cycle = base.Cycle(cycle, rhs, correction, partitioning=part.RedBlack, relaxation_factor=1)
        return cycle

    restriction, prolongation = generate_prolongation_and_restriction([1.0] * problem_size,
                                                                      default_restriction, default_prolongation)
    expression = generate_coarse_grid_correction(restriction, prolongation)

    storages = program_generator.generate_storage(max_level - 1, max_level, grid)
    program_generator.initialize_code_generation(max_level - 1, max_level)
    weight_initialization = program_generator.generate_global_weights(problem_size, 'stencil_weight')
    cycle_function = program_generator.generate_cycle_function(expression, storages, max_level-1, max_level,
                                                               max_level, use_global_weights=True)
    restriction_entries = restriction.entries
    prolongation_entries = prolongation.entries
    start_index = 0
    program = ''
    for i, tmp in enumerate(zip(number_of_restriction_weights, restriction_entries)):
        n = tmp[0]
        entry = tmp[1][i]
        program += program_generator.generate_restriction_operator(entry, max_level, parametrize=True, start_index=start_index)
        start_index += n

    for i, tmp in enumerate(zip(number_of_prolongation_weights, prolongation_entries)):
        n = tmp[0]
        entry = tmp[1][i]
        program += program_generator.generate_prolongation_operator(entry, max_level-1, parametrize=True, start_index=start_index)
        start_index += n

    cycle_function += program

    program_generator.generate_l3_file(max_level - 1, max_level, weight_initialization + cycle_function,
                                       include_restriction=False, include_prolongation=False)

    def evaluate(weights):
        output_path = program_generator._output_path_generated
        program_generator.generate_global_weight_initializations(output_path, weights, name='stencil_weight')
        program_generator.run_c_compiler(output_path)
        runtime, convergence_factor, _ = program_generator.evaluate(output_path, infinity=1e300,
                                                                    number_of_samples=1)
        program_generator.restore_global_initializations(output_path)
        return convergence_factor,

    creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
    creator.create("InterpolationWeights", list, fitness=creator.FitnessMin)
    toolbox = deap.base.Toolbox()
    toolbox.register("evaluate", evaluate)
    lambda_ = int(round((4 + 3 * log(problem_size)) * 2))
    center = 1.0 * len(grid) * 2 / problem_size
    strategy = cma.Strategy(centroid=[center] * problem_size, sigma=(center / 2), lambda_=lambda_)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    toolbox.register("generate", strategy.generate, creator.InterpolationWeights)
    toolbox.register("update", strategy.update)
    hof = tools.HallOfFame(1)
    if program_generator.run_exastencils_compiler(knowledge_path=program_generator.knowledge_path_generated,
                                                  settings_path=program_generator.settings_path_generated) != 0:
        raise RuntimeError("Could not initialize code generator for relaxation factor optimization")
    _, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=iterations, halloffame=hof, verbose=True,
                                             stats=stats)
    weights = hof[0]
    return generate_prolongation_and_restriction(weights, restriction, prolongation)
