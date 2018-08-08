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
        strategy = cma.Strategy(centroid=[0.5] * problem_size, sigma=0.25)

        self._toolbox.register("generate", strategy.generate, creator.Weights)
        self._toolbox.register("update", strategy.update)

        def evaluate(weights):
            weighted_expression = transformations.set_weights(expression, weights)
            iteration_matrix = self._gp_optimizer.get_iteration_matrix(weighted_expression, self._gp_optimizer.grid, self._gp_optimizer.rhs)
            spectral_radius = self._gp_optimizer.convergence_evaluator.compute_spectral_radius(iteration_matrix)
            if spectral_radius == 0.0 or spectral_radius >= 1.0:
                return self._gp_optimizer.infinity,
            elif spectral_radius < 1.0:
                return spectral_radius,
            else:
                raise RuntimeError("Spectral radius out of range")
        self._toolbox.register("evaluate", evaluate)
        hof = tools.HallOfFame(1)
        algorithms.eaGenerateUpdate(self._toolbox, ngen=number_of_generations, halloffame=hof, verbose=True)
        return hof[0]
