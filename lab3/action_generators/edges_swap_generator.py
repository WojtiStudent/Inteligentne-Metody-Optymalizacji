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

        if last_action.name == "swapEdgesInsideCycle":
            i_index = solution[last_action.cycle_index].index(last_action.i)
            j_index = solution[last_action.cycle_index].index(last_action.j)

            # vertexes which changed their status (position or neighbours)
            if i_index > j_index:
                i_index, j_index = j_index, i_index
            if last_action.change_type == "after":
                vertexes = solution[last_action.cycle_index][i_index + 1 : j_index + 1]
                vertexes.append(solution[last_action.cycle_index][i_index])
                vertexes.append(solution[last_action.cycle_index][(j_index + 2) % len(solution[last_action.cycle_index])])
            elif last_action.change_type == "before":
                vertexes = solution[last_action.cycle_index][i_index : j_index]
                vertexes.append(solution[last_action.cycle_index][j_index])
                vertexes.append(solution[last_action.cycle_index][(i_index - 1) % len(solution[last_action.cycle_index])])
            
            vertexes_unch = [i for i in solution[last_action.cycle_index] if i not in vertexes]
            
            for i, j in itertools.product(vertexes, solution[last_action.cycle_index]):
                if i == j:
                    continue
                for change_type in ["after", "before"]:
                    delta = calculate_delta(solution[last_action.cycle_index], i, j, distance_matrix, change_type)
                    if change_type == "after":
                        next_i = solution[last_action.cycle_index][(solution[last_action.cycle_index].index(i) + 1) % len(solution[last_action.cycle_index])]
                        next_j = solution[last_action.cycle_index][(solution[last_action.cycle_index].index(j) + 1) % len(solution[last_action.cycle_index])]
                    else:
                        next_i = solution[last_action.cycle_index][(solution[last_action.cycle_index].index(i) - 1) % len(solution[last_action.cycle_index])]
                        next_j = solution[last_action.cycle_index][(solution[last_action.cycle_index].index(j) - 1) % len(solution[last_action.cycle_index])]
                    if delta < 0:
                        actions.append(
                            Action(
                                name="swapEdgesInsideCycle",
                                i=i,
                                j=j,
                                cycle_index=last_action.cycle_index,
                                delta=delta,
                                do=swap_edges_inside_cycle,
                                change_type=change_type,
                                next_i=next_i,
                                next_j=next_j,
                            )
                        )
        else:
            # leff changed nodes - all left nodes
            vertexes_0 = [last_action.j, last_action.next_i, last_action.prev_i]
            vertexes_1 = [last_action.i, last_action.next_j, last_action.prev_j]

            for i,j in itertools.product(vertexes_0, solution[0]):
                if i == j:
                    continue
                for change_type in ["after", "before"]:
                    delta = calculate_delta(solution[0], i, j, distance_matrix, change_type)
                    if change_type == "after":
                        next_i = solution[0][(solution[0].index(i) + 1) % len(solution[0])]
                        next_j = solution[0][(solution[0].index(j) + 1) % len(solution[0])]
                    else:
                        next_i = solution[0][(solution[0].index(i) - 1) % len(solution[0])]
                        next_j = solution[0][(solution[0].index(j) - 1) % len(solution[0])]
                    if delta < 0:
                        actions.append(
                            Action(
                                name="swapEdgesInsideCycle",
                                i=i,
                                j=j,
                                cycle_index=0,
                                delta=delta,
                                do=swap_edges_inside_cycle,
                                change_type=change_type,
                                next_i=next_i,
                                next_j=next_j,
                            )
                        )
            
            for i,j in itertools.product(vertexes_1, solution[1]):
                if i == j:
                    continue
                for change_type in ["after", "before"]:
                    delta = calculate_delta(solution[1], i, j, distance_matrix, change_type)
                    if change_type == "after":
                        next_i = solution[1][(solution[1].index(i) + 1) % len(solution[1])]
                        next_j = solution[1][(solution[1].index(j) + 1) % len(solution[1])]
                    else:
                        next_i = solution[1][(solution[1].index(i) - 1) % len(solution[1])]
                        next_j = solution[1][(solution[1].index(j) - 1) % len(solution[1])]
                    if delta < 0:
                        actions.append(
                            Action(
                                name="swapEdgesInsideCycle",
                                i=i,
                                j=j,
                                cycle_index=1,
                                delta=delta,
                                do=swap_edges_inside_cycle,
                                change_type=change_type,
                                next_i=next_i,
                                next_j=next_j,
                            )
                        )


        return actions
