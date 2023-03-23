import random
import time

import numpy as np

from utils.scoring import get_cycle_length


class RandomWalkSearch:
    def __init__(self, solution_initializer, actions_generators, time_limit=6.8):
        self.solution_initializer = solution_initializer
        self.actions_generators = actions_generators
        self.time_limit = time_limit

    def get_possible_actions(self, solution, distance_matrix) -> np.array:
        possible_actions = []
        for generator in self.actions_generators:
            possible_actions.extend(generator(solution, distance_matrix))
        return possible_actions

    def __call__(self, distance_matrix):
        solution = self.solution_initializer(distance_matrix)
        score = sum([get_cycle_length(distance_matrix, cycle) for cycle in solution])

        start = time.time()
        while time.time() - start < self.time_limit:
            possible_actions = self.get_possible_actions(solution, distance_matrix)
            action = random.choice(possible_actions)
            action.do(
                solution[action.cycle_index]
                if action.cycle_index != -1
                else solution,
                action.i,
                action.j,
            )
            score += action.delta
        return solution
