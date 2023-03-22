import random

import numpy as np

from utils.scoring import get_cycle_length


class GreedySearch:
    def __init__(self, solution_initializer, actions_generators):
        self.solution_initializer = solution_initializer
        self.actions_generators = actions_generators

    def get_possible_actions(self, solution, distance_matrix) -> np.array:
        possible_actions = []
        for generator in self.actions_generators:
            possible_actions.extend(generator(solution, distance_matrix))
        return possible_actions

    def __call__(self, distance_matrix):
        solution = self.solution_initializer(distance_matrix)
        score = sum([get_cycle_length(distance_matrix, cycle) for cycle in solution])

        improved_done = True
        while improved_done:
            improved_done = False
            possible_actions = self.get_possible_actions(solution, distance_matrix)
            random.shuffle(possible_actions)
            for action in possible_actions:
                if action.delta < 0:
                    action.do(
                        solution[action.cycle_index]
                        if action.cycle_index != -1
                        else solution,
                        action.i,
                        action.j,
                    )
                    score += action.delta
                    improved_done = True
                    break
        return solution
