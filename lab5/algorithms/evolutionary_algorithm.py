from copy import deepcopy
import time
import random
import numpy as np

from utils.scoring import get_summary_score

from lab1.greedy_cycle_two_regret_heuristic import tcp_greedy_cycle_two_regret_heuristic
from lab2.solution_initializers import RandomSolutionGenerator
from lab3.algorithms.SteepestSearchWithMemory import SteepestSearchWithMemory



class EvolutionaryAlgorithm:
    def __init__(self, max_time, population_size=20, minimal_difference=40, 
                 vertex_mutation_chance=0.2, patience=300, ls_on_new_solution=False,
                 solution_initializer=RandomSolutionGenerator,
                 heuristic_algorithm=tcp_greedy_cycle_two_regret_heuristic,
                 ls_algorithm=SteepestSearchWithMemory):
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
        new_solution = deepcopy(parent1_solution)

        remaining = []

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
                    new_solution[i_cycle][i] = new_solution[i_cycle][(i + 1) % len(cycle1)] = -1
                    remaining.extend([v1, v2])
        
        # check for vertices without edges
        for i_cycle, cycle in enumerate(new_solution):
            for i in range(1, len(cycle)):
                prev, curr, next = cycle[i - 1], cycle[i], cycle[(i + 1) % len(cycle)]
                if prev == next == -1 and curr != -1:
                    new_solution[i_cycle][i] = -1
                    remaining.append(curr)

        # mutations
        for i_cycle, cycle in enumerate(new_solution):
            for i in range(len(cycle)):
                if cycle[i] != -1 and random.random() < self.vertex_mutation_chance:
                    new_solution[i_cycle][i] = -1
                    remaining.append(cycle[i])
        
        # delete marked vertices
        for i_cycle, cycle in enumerate(new_solution):
            new_solution[i_cycle] = [v for v in cycle if v != -1]

        # sanity check: add random vertex to cycle without vertex
        for i_cycle, cycle in enumerate(new_solution):
            if len(cycle) == 0:
                chosen = random.choice(remaining)
                remaining.remove(chosen)
                new_solution[i_cycle].append(chosen)

        # rebuild cycles with heuristic algorithm
        new_solution = self.heuristic_algorithm(distance_matrix, new_solution)

        return new_solution

    def __call__(self, distance_matrix):
        start = time.time()

        self.n_iters = 0

        # initialize population
        population = []
        for _ in range(self.population_size):
            random_solution = self.solution_initializer(distance_matrix)
            ls_solution = self.ls_algorithm(random_solution)(distance_matrix)
            score = get_summary_score(distance_matrix, ls_solution)
            population.append((score, ls_solution))

        iterations_without_improvement = 0
        best_score, best_solution = min(population, key=lambda x: x[0])
        worst_unit_index = np.argmax([x[0] for x in population])
        worst_score, worst_solution = population[worst_unit_index]

        while time.time() - start < self.max_time:
            self.n_iters += 1

            # select parents
            parent1, parent2 = random.sample(population, 2)
            parent1_solution, parent2_solution = parent1[1], parent2[1]
            
            # recombine
            new_solution = self._recombine(parent1_solution, parent2_solution, distance_matrix)
            if self.ls_on_new_solution:
                new_solution = self.ls_algorithm(new_solution)(distance_matrix)
            new_solution_score = get_summary_score(distance_matrix, new_solution)

            too_similar = any(abs(new_solution_score - score) < self.minimal_difference for score, _ in population)

            # OUR WAY to update population
            if new_solution_score < best_score:
                best_score, best_solution = new_solution_score, new_solution
                population[worst_unit_index] = (new_solution_score, new_solution)
                worst_unit_index = np.argmax([x[0] for x in population])
                worst_score, worst_solution = population[worst_unit_index]
                iterations_without_improvement = 0
            elif not too_similar and new_solution_score < worst_score:
                population[worst_unit_index] = (new_solution_score, new_solution)
                worst_unit_index = np.argmax([x[0] for x in population])
                worst_score, worst_solution = population[worst_unit_index]
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




        
        


