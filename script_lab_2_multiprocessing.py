import os
from functools import partial

import matplotlib as plt
import numpy as np

plt.rcParams.update({"figure.max_open_warning": 0})
import pandas as pd
import tqdm
import time
from multiprocessing import Pool
from typing import Dict, List, Tuple

import utils.io as io

from lab2.solution_initializers import (
    RandomSolutionGenerator,
    TwoRegretSolutionGenerator,
)
from lab2.action_generators import (
    InCycleVerticesSwapGenerator,
    OutCycleVerticesSwapGenerator,
    EdgesSwapGenerator,
)
from lab2.algorithms import GreedySearch, SteepestSearch, RandomWalkSearch
from utils.scoring import get_cycle_length
from utils.visualization import visualize_graph

N_INSTANCES = 100
DATA_DIR = "data"
RESULT_DIR = io.directory("result/lab2")
FILES = ["kroa100.tsp", "krob100.tsp"]
ALGORITHMS = {
    "random_solution_base": GreedySearch(
        solution_initializer=RandomSolutionGenerator(), actions_generators=[]
    ),
    "two_regret_solution_base": GreedySearch(
        solution_initializer=TwoRegretSolutionGenerator(), actions_generators=[]
    ),
    "greedy_search_two_regret_edges": GreedySearch(
        solution_initializer=TwoRegretSolutionGenerator(),
        actions_generators=[OutCycleVerticesSwapGenerator(), EdgesSwapGenerator()],
    ),
    "greedy_search_two_regret_vertices": GreedySearch(
        solution_initializer=TwoRegretSolutionGenerator(),
        actions_generators=[
            OutCycleVerticesSwapGenerator(),
            InCycleVerticesSwapGenerator(),
        ],
    ),
    "greedy_search_random_edges": GreedySearch(
        solution_initializer=RandomSolutionGenerator(),
        actions_generators=[OutCycleVerticesSwapGenerator(), EdgesSwapGenerator()],
    ),
    "greedy_search_random_vertices": GreedySearch(
        solution_initializer=RandomSolutionGenerator(),
        actions_generators=[
            OutCycleVerticesSwapGenerator(),
            InCycleVerticesSwapGenerator(),
        ],
    ),
    "steepest_search_two_regret_edges": SteepestSearch(
        solution_initializer=TwoRegretSolutionGenerator(),
        actions_generators=[OutCycleVerticesSwapGenerator(), EdgesSwapGenerator()],
    ),
    "steepest_search_two_regret_vertices": SteepestSearch(
        solution_initializer=TwoRegretSolutionGenerator(),
        actions_generators=[
            OutCycleVerticesSwapGenerator(),
            InCycleVerticesSwapGenerator(),
        ],
    ),
    "steepest_search_random_edges": SteepestSearch(
        solution_initializer=RandomSolutionGenerator(),
        actions_generators=[OutCycleVerticesSwapGenerator(), EdgesSwapGenerator()],
    ),
    "steepest_search_random_vertices": SteepestSearch(
        solution_initializer=RandomSolutionGenerator(),
        actions_generators=[
            OutCycleVerticesSwapGenerator(),
            InCycleVerticesSwapGenerator(),
        ],
    ),
    "random_walk_search_two_regret_edges": RandomWalkSearch(
        solution_initializer=TwoRegretSolutionGenerator(),
        actions_generators=[OutCycleVerticesSwapGenerator(), EdgesSwapGenerator()],
    ),
    "random_walk_search_two_regret_vertices": RandomWalkSearch(
        solution_initializer=TwoRegretSolutionGenerator(),
        actions_generators=[
            OutCycleVerticesSwapGenerator(),
            InCycleVerticesSwapGenerator(),
        ],
    ),
    "random_walk_search_random_edges": RandomWalkSearch(
        solution_initializer=RandomSolutionGenerator(),
        actions_generators=[OutCycleVerticesSwapGenerator(), EdgesSwapGenerator()],
    ),
    "random_walk_search_random_vertices": RandomWalkSearch(
        solution_initializer=RandomSolutionGenerator(),
        actions_generators=[
            OutCycleVerticesSwapGenerator(),
            InCycleVerticesSwapGenerator(),
        ],
    ),
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

            # Run the algorithm N_INSTANCES times
            with Pool(8) as p:
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
