from typing import Tuple, List
import numpy as np

from utils.algorithm_utils import get_init_nodes


def find_min_value_index(distance_matrix: np.array) -> Tuple[int, int]:
    """
    Find index of minimum value in distance matrix and return its indexes
    :param distance_matrix: matrix of distances between nodes
    :return: indexes of minimum value in distance matrix
    """
    index = np.argmin(distance_matrix)
    return index // distance_matrix.shape[1], index % distance_matrix.shape[1]


def get_next_node(distance_matrix: np.array, visited_nodes: List[List[int]],
                  current_graph_index: int) -> Tuple[int, int]:
    """
    Get next node to insert in graph
    :param distance_matrix: matrix of distances between nodes
    :param visited_nodes: nodes that already visited
    :param current_graph_index: current graph index
    :return: tuple of (index of place where to insert, node to insert)
    """
    possible_nodes_indexes = [x for x in range(len(distance_matrix)) if x not in visited_nodes[0] + visited_nodes[1]]
    processed_distance_matrix = distance_matrix[visited_nodes[current_graph_index]][:, possible_nodes_indexes]

    sum_distance_matrix = processed_distance_matrix.copy()

    for i in range(len(processed_distance_matrix) - 1):
        left_node_index = visited_nodes[current_graph_index][i]
        right_node_index = visited_nodes[current_graph_index][i+1]
        sum_distance_matrix[i] += sum_distance_matrix[i+1] - distance_matrix[left_node_index][right_node_index]

    if len(processed_distance_matrix) == 2:
        sum_distance_matrix = sum_distance_matrix[:1]
    if len(processed_distance_matrix) > 2:
        sum_distance_matrix[-1] += sum_distance_matrix[0]

    index_row, index_col = find_min_value_index(sum_distance_matrix)
    insert_place = index_row + 1

    return insert_place, possible_nodes_indexes[index_col]


def tcp_greedy_cycle(distance_matrix: np.ndarray) -> List[List[int]]:
    """
    Get two graphs with minimum distance between them using greedy algorithm
    :param distance_matrix: matrix of distances between nodes
    :return: list of two graphs with minimum distance in cycles
    """
    n = len(distance_matrix)
    visited_nodes = get_init_nodes(distance_matrix)
    n_visited = 2

    while n_visited < n:
        cur_graph_index = n_visited % 2
        insert_place, next_node = get_next_node(distance_matrix, visited_nodes, cur_graph_index)
        visited_nodes[cur_graph_index].insert(insert_place, next_node)
        n_visited += 1

    return visited_nodes
