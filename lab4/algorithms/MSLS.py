import copy

import numpy as np

from lab3.algorithms.SteepestSearchWithMemory import SteepestSearchWithMemory
from utils.scoring import get_cycle_length


# using SteepestSearchWithMemory as a base class
class MSLS:
    def __init__(self, solution_initializer):
        self.solution_initializer = solution_initializer
        self.n_iters = 100
        
    def __call__(self, distance_matrix):
        best_solution = None
        best_cost = np.inf
        for i in range(self.n_iters):
            ls_algorithm = SteepestSearchWithMemory(self.solution_initializer)
            solution = ls_algorithm(distance_matrix)
            cost = sum([get_cycle_length(distance_matrix, cycle) for cycle in solution])
            if cost < best_cost:
                best_solution = solution
                best_cost = cost
        return best_solution
