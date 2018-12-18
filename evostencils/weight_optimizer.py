from deap import creator, tools, algorithms, gp, cma
from evostencils.expressions import base, multigrid, transformations
import deap


class WeightOptimizer:
    def __init__(self, gp_optimizer):
        creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
        creator.create("Weights", list, fitness=creator.FitnessMin)
        self._toolbox = deap.base.Toolbox()
        self._gp_optimizer = gp_optimizer

    @staticmethod
    def restrict_weights(weights, minimum, maximum):
        for i, w in enumerate(weights):
            if w < minimum:
                weights[i] = minimum
            elif w > maximum:
                weights[i] = maximum

    def optimize(self, expression: base.Expression, problem_size, number_of_generations):
        def evaluate(weights):
            self.restrict_weights(weights, 0.0, 2.0)
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
        strategy = cma.Strategy(centroid=[1.0] * problem_size, sigma=0.5, lambda_=50)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        import numpy
        stats.register("avg", numpy.mean)
        stats.register("std", numpy.std)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        self._toolbox.register("generate", strategy.generate, creator.Weights)
        self._toolbox.register("update", strategy.update)
        hof = tools.HallOfFame(1)
        algorithms.eaGenerateUpdate(self._toolbox, ngen=number_of_generations, halloffame=hof, verbose=True, stats=stats)
        return hof[0]
