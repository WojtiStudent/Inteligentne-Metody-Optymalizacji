import pandas as pd
import numpy as np
import os
import tqdm
from typing import Dict, List

import utils.io as io
import lab1.nearest_insertion as nnwi
import lab1.scoring as scoring

N_INSTANCES = 100
DATA_DIR = 'data'
FILES = ['kroa100.tsp', 'krob100.tsp']
ALGORITHMS = {"Nearest Inserion": nnwi.tcp_nearest_insertion}

if __name__ == "__main__":
    distance_graphs = [io.create_graph(io.load_data(os.path.join(DATA_DIR, f), 6)) for f in FILES]

    results : Dict[str, pd.DataFrame] = {}  # {algorithm_name: pd.DataFrame(alogorithm_results))}
    best_solutions : Dict[str, Dict[str, List[int]]] = {} # {algorithm_name: {file_name: best_solution}}

    # Run each algorithm
    for name, algorithm in ALGORITHMS.items():
        algorithm_results = []
        file_names = []
        best_algorithm_solutions = {}

        # Run the algorithm for each file
        for file_name, distance_graph in zip(FILES, distance_graphs):
            file_names += [file_name] * N_INSTANCES
            min_cycle_length = np.inf
            best_solution = None
            
            # Run the algorithm N_INSTANCES times
            for i in tqdm.tqdm(range(N_INSTANCES), desc=f"{name} {file_name}"):
                cycles = algorithm(distance_graph)
                cycles_length = sum([scoring.get_cycle_length(distance_graph, cycle) for cycle in cycles])
                algorithm_results.append(cycles_length)

                # Save the best solution
                if cycles_length < min_cycle_length:
                    min_cycle_length = cycles_length
                    best_solution = cycles

            best_algorithm_solutions[file_name] = best_solution

        results[name] = pd.DataFrame({'file': file_names, 'cycles_length': algorithm_results})
        best_solutions[name] = best_algorithm_solutions

    # print(results['Nearest Inserion'])
            
            
