import itertools

from lab2.action_generators.action import Action


def swap_vertices_inside_cycle(cycle, i, j):
    i_index = cycle.index(i)
    j_index = cycle.index(j)
    cycle[i_index], cycle[j_index] = cycle[j_index], cycle[i_index]


def calculate_delta(cycle, i, j, distance_matrix):
    i_index, j_index = cycle.index(i), cycle.index(j)
    before_i, after_i = cycle[i_index - 1], cycle[(i_index + 1) % len(cycle)]
    before_j, after_j = cycle[j_index - 1], cycle[(j_index + 1) % len(cycle)]

    if before_j == i:
        return (
            distance_matrix[before_i, j]
            + distance_matrix[after_j, i]
            - distance_matrix[before_i, i]
            - distance_matrix[j, after_j]
        )
    elif after_j == i:
        return (
            distance_matrix[after_i, j]
            + distance_matrix[i, before_j]
            - distance_matrix[before_j, j]
            - distance_matrix[i, after_i]
        )
    else:
        i_delta_part = (
            distance_matrix[before_j, i]
            + distance_matrix[i, after_j]
            - distance_matrix[before_i, i]
            - distance_matrix[i, after_i]
        )
        j_delta_part = (
            distance_matrix[before_i, j]
            + distance_matrix[j, after_i]
            - distance_matrix[before_j, j]
            - distance_matrix[j, after_j]
        )
        return i_delta_part + j_delta_part


class InCycleVerticesSwapGenerator:
    def __call__(self, solution, distance_matrix):
        actions = []
        for cycle_index, cycle in enumerate(solution):
            possible_pairs = itertools.product(cycle, cycle)
            for i, j in possible_pairs:
                if i != j:
                    actions.append(
                        Action(
                            name="swapVerticesInsideCycle",
                            i=i,
                            j=j,
                            cycle_index=cycle_index,
                            delta=calculate_delta(cycle, i, j, distance_matrix),
                            do=swap_vertices_inside_cycle,
                        )
                    )
        return actions
