from deap import creator, tools, algorithms, gp, cma
from evostencils.expressions import base, multigrid, transformations
import deap


class WeightOptimizer:
    def __init__(self, gp_optimizer):
        creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
        creator.create("Weights", list, fitness=creator.FitnessMin)
        self._toolbox = deap.base.Toolbox()
        self._gp_optimizer = gp_optimizer

    def optimize(self, expression: base.Expression, problem_size, number_of_generations):
        def evaluate(weights):
            tail = transformations.set_weights(expression, weights)
            if len(tail) > 0:
                raise RuntimeError("Incorrect number of weights")
            iteration_matrix = transformations.get_iteration_matrix(expression)
            spectral_radius = self._gp_optimizer.convergence_evaluator.compute_spectral_radius(iteration_matrix)
            if spectral_radius == 0.0:
                return self._gp_optimizer.infinity,
            else:
                return spectral_radius,
        self._toolbox.register("evaluate", evaluate)
        parent = creator.Weights([1.0] * problem_size)
        parent.fitness.values = self._toolbox.evaluate(parent)
        strategy = cma.StrategyOnePlusLambda(parent, sigma=1.0, lambda_=10)
        self._toolbox.register("generate", strategy.generate, creator.Weights)
        self._toolbox.register("update", strategy.update)
        hof = tools.HallOfFame(1)
        algorithms.eaGenerateUpdate(self._toolbox, ngen=number_of_generations, halloffame=hof, verbose=False)
        return hof[0]
