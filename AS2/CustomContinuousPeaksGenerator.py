
""" Classes for defining optimization problem objects."""

# Author: Andrew Rollings
# License: BSD 3 clause

import numpy as np

from mlrose_hiive import DiscreteOpt, ContinuousPeaks
from  mlrose_hiive.algorithms.crossovers import one_point_crossover

class CustomContinuousPeaksGenerator:
    @staticmethod
    def generate(seed, size=60, t_pct=0.1):
        np.random.seed(seed)
        fitness = ContinuousPeaks(t_pct=t_pct)
        problem = DiscreteOpt(length=size, fitness_fn=fitness)

        crossover = one_point_crossover.OnePointCrossOver(problem)

        problem = DiscreteOpt(length=size, fitness_fn=fitness, crossover=crossover)
        return problem