import itertools

from lab2.action_generators.action import Action


def swap_vertices_outside_cycle(cycles, i, j):
    i_index = cycles[0].index(i)
    j_index = cycles[1].index(j)
    cycles[0][i_index], cycles[1][j_index] = cycles[1][j_index], cycles[0][i_index]


def calculate_delta(cycles, i, j, distance_matrix):
    i_index, j_index = cycles[0].index(i), cycles[1].index(j)
    before_i, after_i = (
        cycles[0][i_index - 1],
        cycles[0][(i_index + 1) % len(cycles[0])],
    )
    before_j, after_j = (
        cycles[1][j_index - 1],
        cycles[1][(j_index + 1) % len(cycles[1])],
    )

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


class OutCycleVerticesSwapGenerator:
    def __call__(self, solution, distance_matrix):
        actions = []
        possible_pairs = itertools.product(solution[0], solution[1])
        for i, j in possible_pairs:
            actions.append(
                Action(
                    name="swapVerticesOutsideCycle",
                    i=i,
                    j=j,
                    cycle_index=-1,
                    delta=None,
                    do=swap_vertices_outside_cycle,
                    calculate_delta=calculate_delta,
                )
            )
        return actions
