from typing import List
import numpy as np


def get_cycle_length(distance_graph: np.ndarray, cycle: List[int]) -> int:
    """
    Get cycle length
    :param distance_graph: matrix of distances between nodes
    :param cycle: nodes cycle
    :return: cycle length
    """
    cycle_length = 0
    for i in range(len(cycle) - 1):
        cycle_length += distance_graph[cycle[i]][cycle[i + 1]]
    cycle_length += distance_graph[cycle[-1]][cycle[0]]
    return cycle_length

def get_summary_score(distance_graph: np.ndarray, cycles: List[List[int]]) -> int:
    """
    Get summary score of cycles
    :param cycles: list of cycles
    :return: summary score
    """
    summary_score = 0
    for cycle in cycles:
        summary_score += get_cycle_length(distance_graph, cycle)
    return summary_score



