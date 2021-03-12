import numpy as np
from CustomDiscreteOpt import CustomDiscreteOpt
class CustomKnapSackGenerator:
    @staticmethod
    def generate(seed, number_of_items_types=10,
                 max_item_count=5, max_weight_per_item=25,
                 max_value_per_item=10, max_weight_pct=0.6,
                 multiply_by_max_item_count=True,
                 crossover = None,
                 mutator = None):

        np.random.seed(seed)
        weights = 1 + np.random.randint(max_weight_per_item, size=number_of_items_types)
        values = 1 + np.random.randint(max_value_per_item, size=number_of_items_types)

        problem = KnapsackOpt(length=number_of_items_types,
                                           maximize=True, max_val=max_item_count,
                                           weights=weights, values=values,
                                           max_weight_pct=max_weight_pct,
                                           multiply_by_max_item_count=multiply_by_max_item_count,
                                           crossover=crossover,
                                           mutator=mutator)
        return problem
""" Classes for defining optimization problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

import numpy as np

from mlrose_hiive.algorithms.crossovers import UniformCrossOver
from mlrose_hiive.algorithms.mutators import ChangeOneMutator

""" Classes for defining optimization problem objects."""

# Author: Genevieve Hayes (modified by Andrew Rollings)
# License: BSD 3 clause

from mlrose_hiive.fitness.knapsack import Knapsack


class KnapsackOpt(CustomDiscreteOpt):
    def __init__(self, length=None, fitness_fn=None, maximize=True, max_val=2,
                 weights=None, values=None, max_weight_pct=0.35,
                 crossover=None, mutator=None,
                 multiply_by_max_item_count=False):

        if (fitness_fn is None) and (weights is None and values is None):
            raise Exception("""fitness_fn or both weights and"""
                            + """ values must be specified.""")

        if length is None:
            if weights is not None:
                length = len(weights)
            elif values is not None:
                length = len(values)
            elif fitness_fn is not None:
                length = len(fitness_fn.weights)

        self.length = length

        if fitness_fn is None:
            fitness_fn = Knapsack(weights=weights, values=values,
                                  max_weight_pct=max_weight_pct,
                                  max_item_count=max_val,
                                  multiply_by_max_item_count=multiply_by_max_item_count)

        self.max_val = max_val
        crossover = UniformCrossOver(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator
        super().__init__(length, fitness_fn, maximize, max_val, crossover, mutator)
