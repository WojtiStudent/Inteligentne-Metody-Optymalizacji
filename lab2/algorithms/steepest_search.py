import random

import numpy as np

from utils.scoring import get_cycle_length


class SteepestSearch:
    def __init__(self, solution_initializer, actions_generators):
        self.solution_initializer = solution_initializer
        self.actions_generators = actions_generators

    def get_possible_actions(self, solution, distance_matrix) -> np.array:
        possible_actions = []
        for generator in self.actions_generators:
            possible_actions.extend(generator(solution, distance_matrix))
        return possible_actions

    def __call__(self, distance_matrix, solution=None):
        if solution is None:
            solution = self.solution_initializer(distance_matrix)
        score = sum([get_cycle_length(distance_matrix, cycle) for cycle in solution])

        improved_done = True
        while improved_done:
            improved_done = False
            possible_actions = self.get_possible_actions(solution, distance_matrix)
            best_action = None
            best_delta = 0
            for action in possible_actions:
                if action.delta < best_delta:
                    best_delta = action.delta
                    best_action = action
            
            if best_action != None:
                best_action.do(
                        solution[best_action.cycle_index]
                        if best_action.cycle_index != -1
                        else solution,
                        best_action.i,
                        best_action.j,
                    )
                score += best_delta
                improved_done = True
        return solution
