import itertools

from lab3.action_generators.action import Action


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
            delta = calculate_delta(solution, i, j, distance_matrix)
            if delta < 0:
                i_index = solution[0].index(i)
                j_index = solution[1].index(j)
                next_i = solution[0][(i_index + 1) % len(solution[0])]
                next_j = solution[1][(j_index + 1) % len(solution[1])]
                prev_i = solution[0][(i_index - 1) % len(solution[0])]
                prev_j = solution[1][(j_index - 1) % len(solution[1])]
                actions.append(
                    Action(
                        name="swapVerticesOutsideCycle",
                        i=i,
                        j=j,
                        cycle_index=-1,
                        delta=delta,
                        do=swap_vertices_outside_cycle,
                        next_i=next_i,
                        next_j=next_j,
                        prev_i=prev_i,
                        prev_j=prev_j,
                    )
                )
        return actions

    @staticmethod
    def get_new_actions(solution, distance_matrix, last_action):

        actions = []

        if last_action.name == "swapVerticesOutsideCycle":
            # leff changed nodes - all right nodes
            vertexes_0 = [last_action.j, last_action.next_i, last_action.prev_i]
            vertexes_1 = [last_action.i, last_action.next_j, last_action.prev_j]
            
            for i, j in itertools.product(vertexes_0, solution[1]):
                delta = calculate_delta(solution, i, j, distance_matrix)
                if delta < 0:
                    i_index = solution[0].index(i)
                    j_index = solution[1].index(j)
                    next_i = solution[0][(i_index + 1) % len(solution[0])]
                    next_j = solution[1][(j_index + 1) % len(solution[1])]
                    prev_i = solution[0][(i_index - 1) % len(solution[0])]
                    prev_j = solution[1][(j_index - 1) % len(solution[1])]
                    actions.append(
                        Action(
                            name="swapVerticesOutsideCycle",
                            i=i,
                            j=j,
                            cycle_index=-1,
                            delta=delta,
                            do=swap_vertices_outside_cycle,
                            next_i=next_i,
                            next_j=next_j,
                            prev_i=prev_i,
                            prev_j=prev_j,
                        )
                    )

            # left unchanged nodes - right changed nodes
            for i, j in itertools.product(solution[0], vertexes_1):
                if i in vertexes_0:
                    continue
                delta = calculate_delta(solution, i, j, distance_matrix)
                if delta < 0:
                    i_index = solution[0].index(i)
                    j_index = solution[1].index(j)
                    next_i = solution[0][(i_index + 1) % len(solution[0])]
                    next_j = solution[1][(j_index + 1) % len(solution[1])]
                    prev_i = solution[0][(i_index - 1) % len(solution[0])]
                    prev_j = solution[1][(j_index - 1) % len(solution[1])]
                    actions.append(
                        Action(
                            name="swapVerticesOutsideCycle",
                            i=i,
                            j=j,
                            cycle_index=-1,
                            delta=delta,
                            do=swap_vertices_outside_cycle,
                            next_i=next_i,
                            next_j=next_j,
                            prev_i=prev_i,
                            prev_j=prev_j,
                        )
                    )
        else:
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

            if last_action.cycle_index == 0:
                # left changed nodes - all right nodes
                for i, j in itertools.product(vertexes, solution[1]):
                    delta = calculate_delta(solution, i, j, distance_matrix)
                    if delta < 0:
                        i_index = solution[0].index(i)
                        j_index = solution[1].index(j)
                        next_i = solution[0][(i_index + 1) % len(solution[0])]
                        next_j = solution[1][(j_index + 1) % len(solution[1])]
                        prev_i = solution[0][(i_index - 1) % len(solution[0])]
                        prev_j = solution[1][(j_index - 1) % len(solution[1])]
                        actions.append(
                            Action(
                                name="swapVerticesOutsideCycle",
                                i=i,
                                j=j,
                                cycle_index=-1,
                                delta=delta,
                                do=swap_vertices_outside_cycle,
                                next_i=next_i,
                                next_j=next_j,
                                prev_i=prev_i,
                                prev_j=prev_j,
                            )
                        )
                
            else:
                # all left nodes - right changed nodes
                for i, j in itertools.product(solution[0], vertexes):
                    delta = calculate_delta(solution, i, j, distance_matrix)
                    if delta < 0:
                        i_index = solution[0].index(i)
                        j_index = solution[1].index(j)
                        next_i = solution[0][(i_index + 1) % len(solution[0])]
                        next_j = solution[1][(j_index + 1) % len(solution[1])]
                        prev_i = solution[0][(i_index - 1) % len(solution[0])]
                        prev_j = solution[1][(j_index - 1) % len(solution[1])]
                        actions.append(
                            Action(
                                name="swapVerticesOutsideCycle",
                                i=i,
                                j=j,
                                cycle_index=-1,
                                delta=delta,
                                do=swap_vertices_outside_cycle,
                                next_i=next_i,
                                next_j=next_j,
                                prev_i=prev_i,
                                prev_j=prev_j,
                            )
                        )

        return actions
