from deap import creator, tools, algorithms, cma
from evostencils.expressions import base, multigrid as mg, transformations, partitioning as part
import deap
import numpy
from math import log


def reset_status(expression: base.Expression):
    if isinstance(expression, mg.Cycle):
        expression.iteration_matrix = None
        expression.weight_obtained = False
        expression.weight_set = False
        return reset_status(expression.correction)
    elif isinstance(expression, mg.Residual):
        reset_status(expression.rhs)
        reset_status(expression.approximation)
    elif isinstance(expression, base.UnaryExpression) or isinstance(expression, base.Scaling):
        reset_status(expression.operand)
    elif isinstance(expression, base.BinaryExpression):
        reset_status(expression.operand1)
        reset_status(expression.operand2)


def set_weights(expression: base.Expression, weights: list) -> list:
    if isinstance(expression, mg.Cycle):
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
        return set_weights(expression.correction, tail)
    elif isinstance(expression, mg.Residual):
        tail = set_weights(expression.rhs, weights)
        return set_weights(expression.approximation, tail)
    elif isinstance(expression, base.UnaryExpression) or isinstance(expression, base.Scaling):
        return set_weights(expression.operand, weights)
    elif isinstance(expression, base.BinaryExpression):
        tail = set_weights(expression.operand1, weights)
        return set_weights(expression.operand2, tail)
    else:
        return weights


def obtain_weights(expression: base.Expression) -> list:
    weights = []
    if isinstance(expression, mg.Cycle):
        # Hack to change the weights after generation
        # if isinstance(expression.correction, mg.Residual) \
        #         or (isinstance(expression.correction, base.Multiplication)
        #             and part.can_be_partitioned(expression.correction.operand1)) and
        if not expression.weight_obtained:
            weights.append(expression.weight)
            expression.weight_obtained = True
            weights.extend(obtain_weights(expression.correction))
        return weights
    elif isinstance(expression, mg.Residual):
        weights.extend(obtain_weights(expression.rhs))
        weights.extend(obtain_weights(expression.approximation))
        return weights
    elif isinstance(expression, base.UnaryExpression) or isinstance(expression, base.Scaling):
        weights.extend(obtain_weights(expression.operand))
        return weights
    elif isinstance(expression, base.BinaryExpression):
        weights.extend(obtain_weights(expression.operand1))
        weights.extend(obtain_weights(expression.operand2))
        return weights
    else:
        return weights


def restrict_weights(weights, minimum, maximum):
    for i, w in enumerate(weights):
        if w < minimum:
            weights[i] = minimum
        elif w > maximum:
            weights[i] = maximum


class Optimizer:
    def __init__(self, gp_optimizer):
        creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
        creator.create("Weights", list, fitness=creator.FitnessMin)
        self._toolbox = deap.base.Toolbox()
        self._gp_optimizer = gp_optimizer

    def optimize(self, expression: base.Expression, problem_size, generations, base_program=None, storages=None):
        def evaluate(weights):
            tail = set_weights(expression, weights)
            reset_status(expression)
            if len(tail) > 0:
                raise RuntimeError("Incorrect number of weights")
            generator = self._gp_optimizer.program_generator
            if generator is not None and generator.compiler_available and base_program is not None and \
                    storages is not None:
                # evaluation_program = base_program + generator.generate_cycle_function(expression, storages)
                # generator.write_program_to_file(evaluation_program)
                generator.generate_global_weight_initializations(weights)
                _, convergence_factor = generator.evaluate(only_weights_adapted=True)
                generator.restore_global_initializations()
                generator.invalidate_storages(storages)
                return convergence_factor,
            else:
                iteration_matrix = transformations.get_iteration_matrix(expression)
                spectral_radius = self._gp_optimizer.convergence_evaluator.compute_spectral_radius(iteration_matrix)
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
        self._toolbox.register("generate", strategy.generate, creator.Weights)
        self._toolbox.register("update", strategy.update)
        hof = tools.HallOfFame(1)
        algorithms.eaGenerateUpdate(self._toolbox, ngen=generations, halloffame=hof, verbose=True, stats=stats)
        return hof[0]

