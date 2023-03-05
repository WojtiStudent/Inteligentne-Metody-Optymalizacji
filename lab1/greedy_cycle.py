import random
from typing import Tuple, List
import numpy as np


def get_init_nodes(distance_matrix: np.array) -> List[List[int]]:
    n = len(distance_matrix)
    first_start_node = random.randint(0, n - 1)
    second_start_node = distance_matrix[first_start_node].argmax()

    return [[first_start_node], [second_start_node]]


def find_min_value_index(distance_matrix: np.array) -> Tuple[int, int]:
    index = np.argmin(distance_matrix)
    return index // distance_matrix.shape[1], index % distance_matrix.shape[1]


def get_next_node(distance_matrix: np.array, visited_nodes: List[List[int]],
                  current_graph_index: int) -> Tuple[int, int]:
    possible_nodes_indexes = [x for x in range(len(distance_matrix)) if x not in visited_nodes[0] + visited_nodes[1]]
    processed_distance_matrix = distance_matrix[visited_nodes[current_graph_index]][:, possible_nodes_indexes]

    sum_distance_matrix = processed_distance_matrix.copy()

    for i in range(len(processed_distance_matrix) - 1):
        sum_distance_matrix[i] += sum_distance_matrix[i+1]

    if len(processed_distance_matrix) == 2:
        sum_distance_matrix = sum_distance_matrix[:1]
    if len(processed_distance_matrix) > 2:
        sum_distance_matrix[-1] += sum_distance_matrix[0]

    index_row, index_col = find_min_value_index(sum_distance_matrix)
    insert_place = index_row + 1

    return insert_place, possible_nodes_indexes[index_col]


def tcp_greedy_cycle(distance_matrix: np.ndarray) -> List[List[int]]:
    n = len(distance_matrix)
    visited_nodes = get_init_nodes(distance_matrix)
    n_visited = 2

    while n_visited < n:
        cur_graph_index = n_visited % 2
        insert_place, next_node = get_next_node(distance_matrix, visited_nodes, cur_graph_index)
        visited_nodes[cur_graph_index].insert(insert_place, next_node)
        n_visited += 1

    return visited_nodes
