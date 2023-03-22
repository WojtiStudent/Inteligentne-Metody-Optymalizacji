import itertools

from lab2.action_generators.action import Action


def swap_edges_inside_cycle(cycle, i, j):
    i_index = cycle.index(i)
    j_index = cycle.index(j)

    if i_index > j_index:
        i_index, j_index = j_index, i_index
    reversed = cycle[i_index + 1 : j_index + 1]
    reversed.reverse()

    cycle[i_index + 1 : j_index + 1] = reversed


def calculate_delta(cycle, i, j, distance_matrix):
    i_index, j_index = cycle.index(i), cycle.index(j)
    after_i = cycle[(i_index + 1) % len(cycle)]
    after_j = cycle[(j_index + 1) % len(cycle)]

    return (
        distance_matrix[i, j]
        + distance_matrix[after_i, after_j]
        - distance_matrix[i, after_i]
        - distance_matrix[j, after_j]
    )


class EdgesSwapGenerator:
    def __call__(self, solution, distance_matrix):
        actions = []
        for cycle_index, cycle in enumerate(solution):
            possible_pairs = itertools.product(cycle, cycle)
            for i, j in possible_pairs:
                if i != j:
                    actions.append(
                        Action(
                            name="swapEdgesInsideCycle",
                            i=i,
                            j=j,
                            cycle_index=cycle_index,
                            delta=calculate_delta(cycle, i, j, distance_matrix),
                            do=swap_edges_inside_cycle,
                        )
                    )
        return actions
