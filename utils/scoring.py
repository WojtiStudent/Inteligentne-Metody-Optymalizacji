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
