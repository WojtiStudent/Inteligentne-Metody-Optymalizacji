import itertools

from lab3.action_generators.action import Action


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
            for i, j in itertools.combinations(cycle, 2):
                for change_type in ["after", "before"]:
                    delta = calculate_delta(cycle, i, j, distance_matrix, change_type)
                    if change_type == "after":
                        next_i = cycle[(cycle.index(i) + 1) % len(cycle)]
                        next_j = cycle[(cycle.index(j) + 1) % len(cycle)]
                    else:
                        next_i = cycle[(cycle.index(i) - 1) % len(cycle)]
                        next_j = cycle[(cycle.index(j) - 1) % len(cycle)]
                    if i != j and delta < 0:
                        actions.append(
                            Action(
                                name="swapEdgesInsideCycle",
                                i=i,
                                j=j,
                                cycle_index=cycle_index,
                                delta=delta,
                                do=swap_edges_inside_cycle,
                                change_type=change_type,
                                next_i=next_i,
                                next_j=next_j,
                            )
                        )
        return actions

    def get_new_actions(self, solution, distance_matrix, last_action):
        actions = []

        # nodes_to_check = {"after": [last_action.i, last_action.j],
        #                   "before": [last_action.next_i, last_action.next_j]}

        index_i = solution[last_action.cycle_index].index(last_action.i)
        index_next_i = solution[last_action.cycle_index].index(last_action.next_i)
        index_j = solution[last_action.cycle_index].index(last_action.j)
        index_next_j = solution[last_action.cycle_index].index(last_action.next_j)

        nodes_to_check = {"after": solution[last_action.cycle_index][index_i : index_next_i + 1],
                          "before": solution[last_action.cycle_index][index_j : index_next_j + 1]}

        for change_type, node_list in nodes_to_check.items():
            for node in node_list:
                for cycle_node in solution[last_action.cycle_index]:
                    if node != cycle_node:
                        delta = calculate_delta(solution[last_action.cycle_index],
                                                node,
                                                cycle_node,
                                                distance_matrix,
                                                change_type)
                        if delta < 0:
                            actions.append(
                                Action(
                                    name="swapEdgesInsideCycle",
                                    i=node,
                                    j=cycle_node,
                                    cycle_index=last_action.cycle_index,
                                    delta=delta,
                                    do=swap_edges_inside_cycle,
                                    change_type=change_type,
                                    next_i=last_action.next_i,
                                    next_j=last_action.next_j,
                                )
                            )
        return actions
