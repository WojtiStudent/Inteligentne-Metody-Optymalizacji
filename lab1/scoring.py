from typing import List
import numpy as np

def get_cycle_length(distance_graph: np.ndarray, cycle: List[int]) -> int:
    cycle_length = 0
    for i in range(len(cycle) - 1):
        cycle_length += distance_graph[cycle[i]][cycle[i + 1]]
    cycle_length += distance_graph[cycle[-1]][cycle[0]]
    return cycle_length