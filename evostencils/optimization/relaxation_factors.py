from deap import creator, tools, algorithms, cma
from evostencils.expressions import base
import deap
import numpy
from math import log


def reset_status(expression: base.Expression):
    if isinstance(expression, base.Cycle):
        expression.iteration_matrix = None
        expression.weight_obtained = False
        expression.weight_set = False
        return reset_status(expression.correction)
    elif isinstance(expression, base.Residual):
        reset_status(expression.rhs)
        reset_status(expression.approximation)
    elif isinstance(expression, base.UnaryExpression) or isinstance(expression, base.Scaling):
        reset_status(expression.operand)
    elif isinstance(expression, base.BinaryExpression):
        reset_status(expression.operand1)
        reset_status(expression.operand2)


def set_relaxation_factors(expression: base.Expression, weights: list) -> list:
    if isinstance(expression, base.Cycle):
        if expression.iteration_matrix is not None:
            expression.iteration_matrix = None
        # if len(weights) == 0:
        #     raise RuntimeError("Too few weights have been supplied")
        # if isinstance(expression.correction, mg.Residual) \
        #         or (isinstance(expression.correction, base.Multiplication)
        #             and part.can_be_partitioned(expression.correction.operand1)) and \
        if not expression.weight_set:
            head, *tail = weights
            expression._weight = head
            expression.weight_set = True
            expression.global_id = len(tail)
        else:
            tail = weights
        return set_relaxation_factors(expression.correction, tail)
    elif isinstance(expression, base.Residual):
        tail = set_relaxation_factors(expression.rhs, weights)
        return set_relaxation_factors(expression.approximation, tail)
    elif isinstance(expression, base.UnaryExpression) or isinstance(expression, base.Scaling):
        return set_relaxation_factors(expression.operand, weights)
    elif isinstance(expression, base.BinaryExpression):
        tail = set_relaxation_factors(expression.operand1, weights)
        return set_relaxation_factors(expression.operand2, tail)
    else:
        return weights


def obtain_relaxation_factors(expression: base.Expression) -> list:
    weights = []
    if isinstance(expression, base.Cycle):
        # Hack to change the weights after generation
        # if isinstance(expression.correction, mg.Residual) \
        #         or (isinstance(expression.correction, base.Multiplication)
        #             and part.can_be_partitioned(expression.correction.operand1)) and
        if not expression.weight_obtained:
            weights.append(expression.weight)
            expression.weight_obtained = True
            weights.extend(obtain_relaxation_factors(expression.correction))
        return weights
    elif isinstance(expression, base.Residual):
        weights.extend(obtain_relaxation_factors(expression.rhs))
        weights.extend(obtain_relaxation_factors(expression.approximation))
        return weights
    elif isinstance(expression, base.UnaryExpression) or isinstance(expression, base.Scaling):
        weights.extend(obtain_relaxation_factors(expression.operand))
        return weights
    elif isinstance(expression, base.BinaryExpression):
        weights.extend(obtain_relaxation_factors(expression.operand1))
        weights.extend(obtain_relaxation_factors(expression.operand2))
        return weights
    else:
        return weights


def restrict_relaxation_factors(weights, minimum, maximum):
    for i, w in enumerate(weights):
        if w < minimum:
            weights[i] = minimum
        elif w > maximum:
            weights[i] = maximum


class Optimizer:
    def __init__(self, gp_optimizer):
        creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
        creator.create("RelaxationFactors", list, fitness=creator.FitnessMin)
        self._toolbox = deap.base.Toolbox()
        self._gp_optimizer = gp_optimizer

    def optimize(self, expression: base.Expression, problem_size, generations, storages=None):
        def evaluate(weights):
            tail = set_relaxation_factors(expression, weights)
            reset_status(expression)
            if len(tail) > 0:
                raise RuntimeError("Incorrect number of weights")
            program_generator = self._gp_optimizer.program_generator
            if program_generator is not None and program_generator.compiler_available and \
                    storages is not None:
                # evaluation_program = base_program + generator.generate_cycle_function(expression, storages)
                # generator.write_program_to_file(evaluation_program)
                program_generator.generate_global_weight_initializations(weights)
                program_generator.run_c_compiler()
                _, convergence_factor = program_generator.evaluate()
                program_generator.restore_global_initializations()
                return convergence_factor,
            else:
                spectral_radius = self._gp_optimizer.convergence_evaluator.compute_spectral_radius(expression)
                if spectral_radius == 0.0:
                    return self._gp_optimizer.infinity,
                else:
                    return spectral_radius,
        self._toolbox.register("evaluate", evaluate)
        lambda_ = int((4 + 3 * log(problem_size)) * 2)
        strategy = cma.Strategy(centroid=[1.0] * problem_size, sigma=0.3, lambda_=lambda_)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        self._toolbox.register("generate", strategy.generate, creator.RelaxationFactors)
        self._toolbox.register("update", strategy.update)
        hof = tools.HallOfFame(1)
        generator = self._gp_optimizer.program_generator
        generator.run_exastencils_compiler()
        algorithms.eaGenerateUpdate(self._toolbox, ngen=generations, halloffame=hof, verbose=True, stats=stats)
        generator.restore_files()
        return hof[0]
