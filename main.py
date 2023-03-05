import pandas as pd
import numpy as np
import os
import tqdm
from typing import Dict, List

import utils.io as io
import lab1.nearest_insertion as nnwi
import lab1.greedy_cycle as gc
import lab1.scoring as scoring
from utils.visualization import visualize_graph

N_INSTANCES = 100
DATA_DIR = 'data'
RESULT_DIR = io.directory('result')
FILES = ['kroa100.tsp', 'krob100.tsp']
ALGORITHMS = {"Nearest insertion": nnwi.tcp_nearest_insertion, "Greedy Cycle": gc.tcp_greedy_cycle}

if __name__ == "__main__":
    loaded_files = {f: io.load_data(os.path.join(DATA_DIR, f), 6) for f in FILES}
    distance_graphs = {file_name: io.create_graph(loaded_file) for file_name, loaded_file in loaded_files.items()}

    results: Dict[str, pd.DataFrame] = {}  # {algorithm_name: pd.DataFrame(alogorithm_results))}
    best_solutions: Dict[str, Dict[str, List[int]]] = {} # {algorithm_name: {file_name: best_solution}}

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


    visualize_graph(best_solutions['Greedy Cycle']['kroa100.tsp'], loaded_files['kroa100.tsp'],
                    f'{RESULT_DIR}/greedy_cycle_kroa100.png')
    visualize_graph(best_solutions['Greedy Cycle']['krob100.tsp'], loaded_files['krob100.tsp'],
                    f'{RESULT_DIR}/greedy_cycle_krob100.png')
    visualize_graph(best_solutions['Nearest insertion']['kroa100.tsp'], loaded_files['kroa100.tsp'],
                    f'{RESULT_DIR}/nearest_insertion_kroa100.png')
    visualize_graph(best_solutions['Nearest insertion']['krob100.tsp'], loaded_files['krob100.tsp'],
                    f'{RESULT_DIR}/nearest_insertion_krob100.png')
            
