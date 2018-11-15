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
        strategy = cma.Strategy(centroid=[1.0] * problem_size, sigma=1.0)

        self._toolbox.register("generate", strategy.generate, creator.Weights)
        self._toolbox.register("update", strategy.update)

        def evaluate(weights):
            tail = transformations.set_weights(expression, weights)
            if len(tail) > 0:
                raise RuntimeError("Incorrect number of weights")
            spectral_radius = self._gp_optimizer.convergence_evaluator.compute_spectral_radius(expression)
            if spectral_radius == 0.0:
                return self._gp_optimizer.infinity,
            else:
                return spectral_radius,
        self._toolbox.register("evaluate", evaluate)
        hof = tools.HallOfFame(1)
        algorithms.eaGenerateUpdate(self._toolbox, ngen=number_of_generations, halloffame=hof, verbose=False)
        return hof[0]
