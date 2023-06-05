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



def calculate_similarity(solution_1, solution_2, edges=False):
    """
    Calculate similarity between two cycles
    :param cycle_1: first cycle
    :param cycle_2: second cycle
    :param edges: if True, calculate similarity based on edges, otherwise based on vertices
    :return: similarity
    """
    if edges:
        solution_1_edges = set()
        solution_2_edges = set()
        for cycle in solution_1:
            for i in range(len(cycle) - 1):
                solution_1_edges.add((cycle[i], cycle[i + 1]))
            solution_1_edges.add((cycle[-1], cycle[0]))
        for cycle in solution_2:
            for i in range(len(cycle) - 1):
                solution_2_edges.add((cycle[i], cycle[i + 1]))
            solution_2_edges.add((cycle[-1], cycle[0]))
        
        intersection = len(solution_1_edges.intersection(solution_2_edges))
        return intersection
    else:
        solution_1_vertices_1 = set(solution_1[0])
        solution_1_vertices_2 = set(solution_1[1])
        solution_2_vertices_1 = set(solution_2[0])
        solution_2_vertices_2 = set(solution_2[1])

        intersection_1 = len(solution_1_vertices_1.intersection(solution_2_vertices_1))
        intersection_2 = len(solution_1_vertices_2.intersection(solution_2_vertices_2))
        intersection_3 = len(solution_1_vertices_1.intersection(solution_2_vertices_2))
        intersection_4 = len(solution_1_vertices_2.intersection(solution_2_vertices_1))
        
        return max(intersection_1 + intersection_2, intersection_3 + intersection_4)
    
    