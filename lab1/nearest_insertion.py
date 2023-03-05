import random
import numpy as np
from typing import List, Tuple

from utils.algorithm_utils import get_init_nodes


def find_min_in_array(array: np.ndarray, n: int, visited: List[int]) -> Tuple[int, int]:
    """
    Finds the minimum value in the array and returns its index and the position in the cycle on which to place the vertex.
    :param array: numpy array with distances
    :param n: number of vertexes
    :param visited: list of visited vertexes
    :return: index of the vertex with the minimum distance and the position in the cycle on which to place the vertex
    """
    min_value = np.inf
    min_index = -1
    position_in_cycle = -1
    for i, idx in enumerate(visited):
        for j in range(n):
            if array[idx][j] < min_value and j not in visited:
                min_value = array[idx][j]
                min_index = j
                position_in_cycle = i + 1
    return min_index, position_in_cycle


def tcp_nearest_insertion(distance_graph: np.ndarray) -> List[List[int]]:
    """
    Finds a solution to the Travelling Salesman Problem using the Nearest Insertion algorithm.
    :param distance_graph: numpy array with distances
    :return: two lists of vertexes representing two cycles
    """
    n = len(distance_graph)
    visited_lists = get_init_nodes(distance_graph)
    n_visited = 2
    while n_visited < n:
        for i, cycle in enumerate(visited_lists):
            min_idx, pos = find_min_in_array(distance_graph, n, cycle)
            cycle.insert(pos, min_idx)
            n_visited += 1

    return visited_lists


    



