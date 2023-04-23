import copy

import numpy as np

from lab3.action_generators import EdgesSwapGenerator, OutCycleVerticesSwapGenerator


class SteepestSearchWithMemory:
    def __init__(self, solution_initializer):
        self.solution_initializer = solution_initializer
        self.action_generators = [OutCycleVerticesSwapGenerator(),
                                  EdgesSwapGenerator()]

    def get_possible_actions(self, solution, distance_matrix) -> np.array:
        possible_actions = []
        for generator in self.action_generators:
            possible_actions.extend(generator(solution, distance_matrix))
        return sorted(possible_actions, key=lambda action: action.delta)


    @staticmethod
    def get_node_info(node, solution):
        cycle = 0 if node in solution[0] else 1
        return cycle

    def is_doable(self, action, solution):
        i_cycle = self.get_node_info(action.i, solution)
        j_cycle = self.get_node_info(action.j, solution)

        if action.name == "swapVerticesOutsideCycle":
            if i_cycle != j_cycle:
                return "DOABLE"
            else:
                return "NOT_DOABLE"
        elif action.name == "swapEdgesInsideCycle":
            if i_cycle == j_cycle:
                next_i = solution[i_cycle][(solution[i_cycle].index(action.i) + 1) % len(solution[i_cycle])]
                next_j = solution[j_cycle][(solution[j_cycle].index(action.j) + 1) % len(solution[j_cycle])]

                previous_i = solution[i_cycle][(solution[i_cycle].index(action.i) - 1) % len(solution[i_cycle])]
                previous_j = solution[j_cycle][(solution[j_cycle].index(action.j) - 1) % len(solution[j_cycle])]
                
                if action.next_i == next_i and action.next_j == next_j:
                    return "DOABLE"
                elif action.next_i == previous_i and action.next_j == previous_j:
                    return "MAYBE"
                else:
                    return "NOT_DOABLE"
            else:
                return "NOT_DOABLE"

    def get_new_possible_actions(self, solution, distance_matrix, last_action, possible_actions):
        new_possible_actions = []
        if last_action.name == "swapVerticesOutsideCycle":
            new_possible_actions += self.action_generators[0].get_new_actions(solution, distance_matrix, last_action)

            for action in possible_actions:
                new_action = copy.deepcopy(action)
                if new_action.name == "swapEdgesInsideCycle":
                    if new_action.i == last_action.i:
                        new_action.i = last_action.j
                    elif new_action.i == last_action.j:
                        new_action.i = last_action.i
                    elif new_action.j == last_action.i:
                        new_action.j = last_action.j
                    elif new_action.j == last_action.j:
                        new_action.j = last_action.i

                    new_possible_actions.append(new_action)


        elif last_action.name == "swapEdgesInsideCycle":
            new_possible_actions = self.action_generators[1].get_new_actions(solution, distance_matrix, last_action)
        return new_possible_actions

    def __call__(self, distance_matrix):
        solution = self.solution_initializer(distance_matrix)
        possible_actions = self.get_possible_actions(solution, distance_matrix)

        while possible_actions:
            print(solution)
            print(possible_actions[:3])


            last_action = None
            for i, action in enumerate(possible_actions):
                is_doable_flag = self.is_doable(action, solution)
                if is_doable_flag == "DOABLE":
                    action.do(solution[action.cycle_index] if action.cycle_index != -1 else solution,
                              action.i,
                              action.j)
                    possible_actions.pop(i)
                    last_action = action
                    break
                elif is_doable_flag == "NOT_DOABLE":
                    possible_actions.pop(i)
                elif is_doable_flag == "MAYBE":
                    pass

            if last_action:
                possible_actions += self.get_new_possible_actions(solution,
                                                                  distance_matrix,
                                                                  last_action,
                                                                  possible_actions)
                possible_actions = sorted(possible_actions, key=lambda action: action.delta)
            else:
                break

        return solution
