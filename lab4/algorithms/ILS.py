import copy

import numpy as np
import math
import random

from time import time

from lab3.algorithms.SteepestSearchWithMemory import SteepestSearchWithMemory
from lab4.actions.action import Action
from lab1.greedy_cycle_two_regret_heuristic import tcp_greedy_cycle_two_regret_heuristic
from utils.scoring import get_cycle_length

class ILS:
    def __init__(self, solution_initializer, lifespan, no=1, destroy_ratio=0.2):
        self.solution_initializer = solution_initializer
        self.lifespan = lifespan
        self.no = no
        self.destroy_ratio = destroy_ratio

    def _destroy(self, solution):
        n = len(solution[0]) + len(solution[1])
        n_to_destroy = int(n * self.destroy_ratio)
        n_sequences_to_destroy = random.randint(math.floor(math.sqrt(n_to_destroy)), n_to_destroy//2)
        seqence_length = n_to_destroy // n_sequences_to_destroy
        rest = n_to_destroy % n_sequences_to_destroy

        for i in range(n_sequences_to_destroy):
            cycle_index = random.randint(0, 1)
            start_idx = random.randint(0, len(solution[cycle_index])-1)
            for _ in range(seqence_length):
                solution[cycle_index].pop(start_idx % len(solution[cycle_index]))
            if rest > 0:
                solution[cycle_index].pop(start_idx % len(solution[cycle_index]))
                rest -= 1


    def __call__(self, distance_matrix):
        best_solution = self.solution_initializer(distance_matrix)
        tmp_solution = copy.deepcopy(best_solution)
        best_cost = sum([get_cycle_length(distance_matrix, cycle) for cycle in best_solution])
        ns = [len(cycle) for cycle in best_solution]
        start = time()

        # ILS1
        if self.no == 1:
            while time() - start < self.lifespan:
                action = Action(tmp_solution)
                action.do(tmp_solution if action.name == "swapVerticesOutsideCycle" else tmp_solution[action.cycle_index])
                
                cost = sum([get_cycle_length(distance_matrix, cycle) for cycle in tmp_solution])
                if cost < best_cost:
                    best_solution = copy.deepcopy(tmp_solution)
                    best_cost = cost
                else:
                    tmp_solution = copy.deepcopy(best_solution)
        
        # ILS2
        elif self.no == 2:
            while time() - start < self.lifespan:
                self._destroy(tmp_solution)
                tmp_solution = tcp_greedy_cycle_two_regret_heuristic(distance_matrix, tmp_solution)
                cost = sum([get_cycle_length(distance_matrix, cycle) for cycle in tmp_solution])
                if cost < best_cost:
                    best_solution = copy.deepcopy(tmp_solution)
                    best_cost = cost
                else:
                    tmp_solution = copy.deepcopy(best_solution)

        return best_solution        



