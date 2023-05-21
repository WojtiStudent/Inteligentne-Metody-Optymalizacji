import os
import matplotlib as plt

plt.rcParams.update({"figure.max_open_warning": 0})
import numpy as np
import pandas as pd
import tqdm
import time
from typing import Dict, List, Tuple

import utils.io as io

from lab2.solution_initializers import (
    TwoRegretSolutionGenerator, RandomSolutionGenerator,
)

from utils.scoring import get_cycle_length
from utils.visualization import visualize_graph

from lab5.algorithms.evolutionary_algorithm import EvolutionaryAlgorithm

N_INSTANCES = 1
DATA_DIR = "data"
RESULT_DIR = io.directory("result/lab5")
FILES = ["kroa200.tsp", "krob200.tsp"]

SOLUTION_INITIALIZER = RandomSolutionGenerator()
EVO_LIFE_SPAN = 100 # 30 for two regret | 220 for random

ALGORITHMS = {
    "EVO": EvolutionaryAlgorithm(max_time=EVO_LIFE_SPAN, solution_initializer=SOLUTION_INITIALIZER),
    "EVO+LS": EvolutionaryAlgorithm(max_time=EVO_LIFE_SPAN, solution_initializer=SOLUTION_INITIALIZER, ls_on_new_solution=True),
}

if __name__ == "__main__":
    loaded_files = {f: io.load_data(os.path.join(DATA_DIR, f), 6) for f in FILES}
    distance_graphs = {
        file_name: io.create_graph(loaded_file)
        for file_name, loaded_file in loaded_files.items()
    }

    results: Dict[
        str, pd.DataFrame
    ] = {}  # {algorithm_name: pd.DataFrame(alogorithm_results))}
    best_solutions: Dict[
        str, Dict[str, List[int]]
    ] = {}  # {algorithm_name: {file_name: best_solution}}

    times: List[Tuple[str, str]] = []

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

            # Run the algorithm N_INSTANCES times
            for i in tqdm.tqdm(range(N_INSTANCES), desc=f"{name} {file_name}"):
                start = time.time()
                cycles = algorithm(distance_graph)
                local_times.append(time.time() - start)
                cycles_length = sum(
                    [get_cycle_length(distance_graph, cycle) for cycle in cycles]
                )
                algorithm_results.append(cycles_length)

                # Save the best solution
                if cycles_length < min_cycle_length:
                    min_cycle_length = cycles_length
                    best_solution = cycles

            times.append(
                (
                    f"{name}_{file_name}",
                    f"{round(np.mean(local_times), 3)}({round(min(local_times), 3)} - {round(max(local_times), 3)})",
                )
            )
            best_algorithm_solutions[file_name] = best_solution
            visualize_graph(
                best_solution,
                loaded_files[file_name],
                f"{RESULT_DIR}/{'_'.join(name.split())}_{file_name.split('.')[0]}_{min_cycle_length}.png",
            )

        results[name] = pd.DataFrame(
            {"file": file_names, "cycles_length": algorithm_results}
        )
        best_solutions[name] = best_algorithm_solutions

    for name, result in results.items():
        grouped_df = result.groupby("file")

        print(name)
        for key, item in grouped_df:
            mean = grouped_df.get_group(key)["cycles_length"].mean()
            min_len = grouped_df.get_group(key)["cycles_length"].min()
            max_len = grouped_df.get_group(key)["cycles_length"].max()
            print(f"\tFile: {key}  | Score: {mean}({min_len} - {max_len})")

    for i in times:
        tmp = i[0].split("_")
        file_name = tmp[-1]
        algorithm_name = "_".join(tmp[:-1])
        print(algorithm_name)
        print(f"\tFile: {file_name}  | Time: {i[1]}")
