from functools import partial
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt

from lab2.algorithms.greedy_search import GreedySearch

plt.rcParams.update({"figure.max_open_warning": 0})
import numpy as np
import pandas as pd
import tqdm
import time
from typing import Dict, List, Tuple

import utils.io as io

from lab2.solution_initializers import (
    RandomSolutionGenerator,
)

from utils.scoring import get_cycle_length, calculate_similarity
from utils.visualization import visualize_graph

    
from lab2.action_generators import (
    InCycleVerticesSwapGenerator,
    OutCycleVerticesSwapGenerator,
    EdgesSwapGenerator,
)

N_INSTANCES = 1000
DATA_DIR = "data"
RESULT_DIR = io.directory("result/lab6/correlation")
FILES = ["kroa100.tsp", "krob100.tsp"]

SOLUTION_INITIALIZER = RandomSolutionGenerator()

ALGORITHMS = {
   
    "greedy_search_random_edges": GreedySearch(
        solution_initializer=RandomSolutionGenerator(),
        actions_generators=[OutCycleVerticesSwapGenerator(), EdgesSwapGenerator()],
    ),
   
}


def run_algorithm(i, algorithm, distance_graph):
    cycles = algorithm(distance_graph)
    cycles_length = sum([get_cycle_length(distance_graph, cycle) for cycle in cycles])
    return {"cycles": cycles, "length": cycles_length}


if __name__ == "__main__":
    loaded_files = {f: io.load_data(os.path.join(DATA_DIR, f), 6) for f in FILES}
    distance_graphs = {
        file_name: io.create_graph(loaded_file)
        for file_name, loaded_file in loaded_files.items()
    }

   
    # Run each algorithm
    for name, algorithm in ALGORITHMS.items():
        algorithm_results = []
        file_names = []
        best_algorithm_solutions = {}

        # Run the algorithm for each file
        for file_name, distance_graph in distance_graphs.items():
            file_names += [file_name] * N_INSTANCES
            min_cycle_length = np.inf
            best_solution = None
            local_times = []


            cycles_list = []
            cycle_lengths = []
            similarity_to_optimal = []
            similarity_to_mean = []
            # Run the algorithm N_INSTANCES times
            with Pool(4) as p:
                cycles = list(
                    tqdm.tqdm(
                        p.imap(
                            partial(
                                run_algorithm,
                                algorithm=algorithm,
                                distance_graph=distance_graph,
                            ),
                            range(N_INSTANCES),
                        ),
                        desc=f"{name} {file_name}",
                    )
                )

            similarity_table_vertices = np.zeros((N_INSTANCES, N_INSTANCES))
            similarity_table_edges = np.zeros((N_INSTANCES, N_INSTANCES))

            # Calculate similarity between each pair of solutions
            for i in range(N_INSTANCES):
                for j in range(N_INSTANCES):
                    if i != j:
                        similarity_table_vertices[i, j] = calculate_similarity(cycles[i]["cycles"], cycles[j]["cycles"])
                        similarity_table_edges[i, j] = calculate_similarity(cycles[i]["cycles"], cycles[j]["cycles"], edges=True)

            
            # ===================== Vertices =====================
            similarity_to_mean = np.mean(similarity_table_vertices, axis=1)

            best_solution_index = np.argmin([cycle["length"] for cycle in cycles])
            similarity_to_optimal = similarity_table_vertices[best_solution_index, :]

            scores = [cycle["length"] for cycle in cycles]  


            # calculate correlation
            correlation = np.corrcoef(scores, similarity_to_mean)[0, 1]          
            plt.scatter(scores, similarity_to_mean)
            plt.xlabel("Score")
            plt.ylabel("Similarity to mean")
            
            plt.grid(True)
            plt.title(f"{file_name} | vertices | similarity to mean | correlation: {round(correlation, 3)}")
            plt.savefig(os.path.join(RESULT_DIR, f"{file_name}_score_similarity_to_mean.png"))
            plt.clf()


            
            scores_without_best = scores.copy()
            scores_without_best.remove(min(scores_without_best))
            similarity_to_optimal_without_best = similarity_to_optimal.copy()
            similarity_to_optimal_without_best = np.delete(similarity_to_optimal_without_best, np.argmin(scores))
            
            correlation = np.corrcoef(scores_without_best, similarity_to_optimal_without_best)[0, 1]
            plt.scatter(scores_without_best, similarity_to_optimal_without_best)
            plt.xlabel("Score")
            plt.ylabel("Similarity to optimal")
            plt.title(f"{file_name} | vertices | similarity to optimal | correlation: {round(correlation, 3)}")
            plt.grid(True)
            plt.savefig(os.path.join(RESULT_DIR, f"{file_name}_score_similarity_to_optimal.png"))
            plt.clf()

            # ===================== Edges =====================
            similarity_to_mean = np.mean(similarity_table_edges, axis=1)

            best_solution_index = np.argmin([cycle["length"] for cycle in cycles])
            similarity_to_optimal = similarity_table_edges[best_solution_index, :]

            scores = [cycle["length"] for cycle in cycles]

            # calculate correlation
            correlation = np.corrcoef(scores, similarity_to_mean)[0, 1]
            plt.scatter(scores, similarity_to_mean)
            plt.xlabel("Score")
            plt.ylabel("Similarity to mean")
            plt.title(f"{file_name} | edges | similarity to mean | correlation: {round(correlation, 3)}")
            plt.grid(True)
            plt.savefig(os.path.join(RESULT_DIR, f"{file_name}_score_similarity_to_mean_edges.png"))
            plt.clf()


            scores_without_best = scores.copy()
            scores_without_best.remove(min(scores_without_best))
            similarity_to_optimal_without_best = similarity_to_optimal.copy()
            similarity_to_optimal_without_best = np.delete(similarity_to_optimal_without_best, np.argmin(scores))
            
            correlation = np.corrcoef(scores_without_best, similarity_to_optimal_without_best)[0, 1]
            plt.scatter(scores_without_best, similarity_to_optimal_without_best)
            plt.xlabel("Score")
            plt.ylabel("Similarity to optimal")
            plt.title(f"{file_name} | edges | similarity to optimal | correlation: {round(correlation, 3)}")
            plt.grid(True)
            plt.savefig(os.path.join(RESULT_DIR, f"{file_name}_score_similarity_to_optimal_edges.png"))
            plt.clf()
