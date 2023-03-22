import random

from lab1.greedy_cycle_two_regret_heuristic import tcp_greedy_cycle_two_regret_heuristic


class RandomSolutionGenerator:
    def __call__(self, distance_graph):
        nodes = list(range(len(distance_graph)))
        random.shuffle(nodes)
        return [nodes[: len(nodes) // 2], nodes[len(nodes) // 2 :]]


class TwoRegretSolutionGenerator:
    def __call__(self, distance_graph):
        return tcp_greedy_cycle_two_regret_heuristic(distance_graph)
