import random
import numpy as np
from typing import List, Tuple

from utils.algorithm_utils import get_init_nodes


def find_min_in_array(array: np.ndarray, n: int, cycle: List[int], visited: List[int]) -> Tuple[int, int]:
    """
    Finds the minimum value in the array and returns its index and the position in the cycle on which to place the vertex.
    :param array: numpy array with distances
    :param n: number of vertexes
    :param cycle: list of vertexes in the cycle
    :param visited: list of visited vertexes
    :return: index of the vertex with the minimum distance and the position in the cycle on which to place the vertex
    """
    min_value = np.inf
    new_vertex = None
    position_in_cycle = None
    for i, vertex in enumerate(cycle):
        for j in range(n):
            if array[vertex][j] < min_value and j not in visited:
                min_value = array[vertex][j]
                new_vertex = j
                position_in_cycle = i
    return new_vertex, position_in_cycle


def tcp_nearest_insertion(distance_graph: np.ndarray) -> List[List[int]]:
    """
    Finds a solution to the Travelling Salesman Problem using the Nearest Insertion algorithm.
    :param distance_graph: numpy array with distances
    :return: two lists of vertexes representing two cycles
    """
    n = len(distance_graph)
    cycles = get_init_nodes(distance_graph)
    visited = [cycles[0][0], cycles[1][0]]
    n_visited = 2
    while n_visited < n:
        for i, cycle in enumerate(cycles):
            new_vertex, pos = find_min_in_array(distance_graph, n, cycle, visited)
            cycle.insert(pos, new_vertex)
            visited.append(new_vertex)
            n_visited += 1
    return cycles


    



