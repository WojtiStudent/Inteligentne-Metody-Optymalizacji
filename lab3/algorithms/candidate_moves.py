import numpy as np
from lab2.action_generators import Action
import lab2.action_generators.out_cycle_vertices_swap_generator as out_cycle_vertices
import lab2.action_generators.edges_swap_generator as edges_swap


class CandidateMoves:
    def __init__(self, solution_initializer):
        self.solution_initializer = solution_initializer

    @staticmethod
    def get_nearest_neighbours(distance_matrix):
        nearest_neighbours = []
        for i in range(len(distance_matrix)):
            nearest_neighbours.append(np.argsort(distance_matrix[i])[1:11])
        return nearest_neighbours

    @staticmethod
    def get_node_info(node, solution):
        cycle = 0 if node in solution[0] else 1
        return cycle

    def __call__(self, distance_matrix):
        solution = self.solution_initializer(distance_matrix)
        nearest_neighbours = self.get_nearest_neighbours(distance_matrix)

        while True:
            best_action = None
            for node in range(len(nearest_neighbours)):
                for neighbour in nearest_neighbours[node]:
                    node_cycle = self.get_node_info(node, solution)
                    neighbour_cycle = self.get_node_info(neighbour, solution)

                    if node_cycle == neighbour_cycle:
                        delta_change_after = edges_swap.calculate_delta(
                            solution[node_cycle],
                            node,
                            neighbour,
                            distance_matrix,
                            "after",
                        )
                        delta_change_before = edges_swap.calculate_delta(
                            solution[node_cycle],
                            node,
                            neighbour,
                            distance_matrix,
                            "before",
                        )

                        if delta_change_after < delta_change_before:
                            action = Action(
                                name="swapEdges",
                                i=node,
                                j=neighbour,
                                cycle_index=node_cycle,
                                delta=delta_change_after,
                                do=edges_swap.swap_edges_inside_cycle,
                                change_type="after",
                            )
                        else:
                            action = Action(
                                name="swapEdges",
                                i=node,
                                j=neighbour,
                                cycle_index=node_cycle,
                                delta=delta_change_before,
                                do=edges_swap.swap_edges_inside_cycle,
                                change_type="before",
                            )
                    else:
                        if node_cycle != 0:
                            node, neighbour = neighbour, node
                        action = Action(
                            name="swapVerticesOutsideCycle",
                            i=node,
                            j=neighbour,
                            cycle_index=-1,
                            delta=out_cycle_vertices.calculate_delta(
                                solution, node, neighbour, distance_matrix
                            ),
                            do=out_cycle_vertices.swap_vertices_outside_cycle,
                        )
                    if best_action is None or action.delta < best_action.delta:
                        best_action = action

            if best_action is not None and best_action.delta < 0:
                if best_action.name == "swapEdges":
                    best_action.do(
                        solution[best_action.cycle_index],
                        best_action.i,
                        best_action.j,
                        best_action.change_type,
                    )
                else:
                    best_action.do(solution, best_action.i, best_action.j)
            else:
                break
        return solution
