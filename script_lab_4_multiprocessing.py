import os
from functools import partial
from multiprocessing import Pool

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

from lab4.algorithms.MSLS import MSLS
from lab4.algorithms.ILS import ILS

N_INSTANCES = 8
DATA_DIR = "data"
RESULT_DIR = io.directory("result/lab4")
FILES = ["krob200.tsp"]

SOLUTION_INITIALIZER = RandomSolutionGenerator()
ILS_LIFE_SPAN = 330 # 28 for two regret | 220 for random

ALGORITHMS = {
    "MSLS": MSLS(solution_initializer=SOLUTION_INITIALIZER),
    "ILS1": ILS(solution_initializer=SOLUTION_INITIALIZER, lifespan=ILS_LIFE_SPAN, no=1),
    "ILS2": ILS(solution_initializer=SOLUTION_INITIALIZER, lifespan=ILS_LIFE_SPAN, no=2, with_local_search=True),
    "ILS2a": ILS(solution_initializer=SOLUTION_INITIALIZER, lifespan=ILS_LIFE_SPAN, no=2, with_local_search=False),
}


def run_algorithm(i, algorithm, distance_graph):
    start = time.time()
    cycles = algorithm(distance_graph)
    cycles_length = sum([get_cycle_length(distance_graph, cycle) for cycle in cycles])
    end = time.time() - start
    return {"cycles": cycles, "length": cycles_length, "time": end}


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
            local_times = []

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

            best_cycle = min(cycles, key=lambda cycle: cycle["length"])
            best_solution = best_cycle["cycles"]
            best_cycle_length = best_cycle["length"]

            algorithm_results += [cycle["length"] for cycle in cycles]
            min_cycle_length = min(algorithm_results)

            algorithm_times = [cycle["time"] for cycle in cycles]

            times.append(
                (
                    f"{name}_{file_name}",
                    f"{round(np.mean(algorithm_times), 3)}({round(min(algorithm_times), 3)} - {round(max(algorithm_times), 3)})",
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
