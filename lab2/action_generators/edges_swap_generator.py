import itertools

from lab2.action_generators.action import Action


def swap_edges_inside_cycle(cycle, i, j, change_type="after"):
    i_index = cycle.index(i)
    j_index = cycle.index(j)

    if i_index > j_index:
        i_index, j_index = j_index, i_index

    if change_type == "after":
        reversed = cycle[i_index + 1 : j_index + 1]
        reversed.reverse()
        cycle[i_index + 1 : j_index + 1] = reversed
    elif change_type == "before":
        reversed = cycle[i_index : j_index]
        reversed.reverse()
        cycle[i_index : j_index] = reversed


def calculate_delta(cycle, i, j, distance_matrix, change_type="after"):
    i_index, j_index = cycle.index(i), cycle.index(j)

    if change_type == "after":
        next_i = cycle[(i_index + 1) % len(cycle)]
        next_j = cycle[(j_index + 1) % len(cycle)]
    else:
        next_i = cycle[(i_index - 1) % len(cycle)]
        next_j = cycle[(j_index - 1) % len(cycle)]

    return (
        distance_matrix[i, j]
        + distance_matrix[next_i, next_j]
        - distance_matrix[i, next_i]
        - distance_matrix[j, next_j]
    )


class EdgesSwapGenerator:
    def __call__(self, solution, distance_matrix):
        actions = []
        for cycle_index, cycle in enumerate(solution):
            possible_pairs = itertools.combinations(cycle, r=2 ) #itertools.product(cycle, cycle)
            for i, j in possible_pairs:
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
