from copy import deepcopy
import time
import random
import numpy as np

from utils.scoring import get_cycle_length




class EvolutionaryAlgorithm:
    def __init__(self, population_size, max_time, minimal_difference, 
                 vertex_mutation_chance, patience, ls_on_new_solution,
                 solution_initializer, heuristic_algorithm, ls_algorithm):
        self.population_size = population_size
        self.max_time = max_time
        self.minimal_difference = minimal_difference
        self.vertex_mutation_chance = vertex_mutation_chance
        self.patience = patience
        self.ls_on_new_solution = ls_on_new_solution
        self.solution_initializer = solution_initializer
        self.heuristic_algorithm = heuristic_algorithm
        self.ls_algorithm = ls_algorithm

    def _recombine(self, parent1_solution, parent2_solution, distance_matrix):
        remaining = []
        new_solution = deepcopy(parent1_solution)

        # check for the same edges in both parent solutions
        for i_cycle, cycle1 in enumerate(parent1_solution):
            for i in range(len(cycle1)):
                v1, v2 = cycle1[i], cycle1[(i + 1) % len(cycle1)]

                in_both_cycles = False
                for cycle2 in parent2_solution:
                    if v1 in cycle2 and v2 in cycle2:
                        i1, i2 = cycle2.index(v1), cycle2.index(v2)
                        # if vertexes are next to each other
                        if abs(i1 - i2) == 1 or abs(i1 - i2) == len(cycle2) - 1: 
                            in_both_cycles = True
                            break
                if not in_both_cycles:
                    remaining.extend([v1, v2])
                    new_solution[i_cycle][i] = new_solution[i_cycle][(i + 1) % len(cycle1)] = -1
        
        # check for vertices without edges
        for i_cycle, cycle in enumerate(new_solution):
            for i in range(1, len(cycle)):
                prev, curr, next = cycle[i - 1], cycle[i], cycle[(i + 1) % len(cycle)]
                if prev == next == -1 and curr != -1:
                    remaining.append(curr)
                    new_solution[i_cycle][i] = -1

        # mutations
        for i_cycle, cycle in enumerate(new_solution):
            for i in range(len(cycle)):
                if cycle[i] != -1 and random.random() < self.vertex_mutation_chance:
                    remaining.append(object)
                    new_solution[i_cycle][i] = -1
        
        # delete marked vertices
        for i_cycle, cycle in enumerate(new_solution):
            new_solution[i_cycle] = [v for v in cycle if v != -1]

        # rebuild cycles with heuristic algorithm
        # TODO: check, remaining is probably not needed here
        new_solution = self.heuristic_algorithm(distance_matrix, new_solution)

        return new_solution

    def __call__(self, distance_matrix):
        start = time.time()

        # initialize population
        population = []
        for _ in range(self.population_size):
            random_solution = self.solution_initializer(distance_matrix)
            ls_solution = self.ls_algorithm(distance_matrix, random_solution)
            score = get_cycle_length(distance_matrix, ls_solution)
            population.append((score, ls_solution))

        iterations_without_improvement = 0
        best_score, best_solution = *min(population, key=lambda x: x[0])
        worst_unit_index = np.argmax([x[0] for x in population])
        worst_score, worst_solution = *population[worst_unit_index]

        while tiem.time() - start < self.max_time:
            # select parents
            parent1, parent2 = random.sample(population, 2)
            parent1_solution, parent2_solution = parent1[1], parent2[1]
            
            # recombine
            new_solution = self._recombine(parent1_solution, parent2_solution, distance_matrix)
            if self.ls_on_new_solution:
                new_solution = self.ls_algorithm(distance_matrix, new_solution)
            new_solution_score = get_cycle_length(distance_matrix, new_solution)

            too_similar = any(abs(new_solution_score - score) < self.minimal_difference for score, _ in population)

            # OUR WAY to update population
            if new_solution_score < best_score:
                best_score, best_solution = new_solution_score, new_solution
                population[worst_unit_index] = (new_solution_score, new_solution)
                worst_unit_index = np.argmax([x[0] for x in population])
                worst_score, worst_solution = *population[worst_unit_index]
                iterations_without_improvement = 0
            elif not too_similar and new_solution_score < worst_score:
                population[worst_unit_index] = (new_solution_score, new_solution)
                worst_unit_index = np.argmax([x[0] for x in population])
                worst_score, worst_solution = *population[worst_unit_index]
                iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1

            ## dabliu ei dabliu ei dabliu ei WAY to update population
            # if new_solution_score < best_score:
            #     best_unit_index = np.argmin([x[0] for x in population])
            #     best_score, best_solution = new_solution_score, new_solution
            #     population[best_unit_index] = (new_solution_score, new_solution)
            #     iterations_without_improvement = 0
            # elif not too_similar and new_solution_score < worst_score:
            #     worst_unit_index = np.argmax([x[0] for x in population])
            #     worst_score, worst_solution = new_solution_score, new_solution
            #     population[worst_unit_index] = (new_solution_score, new_solution)
            #     iterations_without_improvement += 1
            # else:
            #     iterations_without_improvement += 1

            if iterations_without_improvement == self.patience:
                break
        
        return best_solution




        
        


