from typing import List
import random
import numpy as np


def get_init_nodes(distance_matrix: np.array) -> List[List[int]]:
    """
    Get two nodes with the maximum distance between them
    :param distance_matrix: matrix of distances between nodes
    :return: initial nodes in two graphs
    """
    n = len(distance_matrix)
    first_start_node = random.randint(0, n - 1)
    second_start_node = distance_matrix[first_start_node].argmax()

    return [[first_start_node], [second_start_node]]
