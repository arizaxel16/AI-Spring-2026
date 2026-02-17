from tsp import TSP
from tsp_search_builders import get_genetic_search_for_tsp, get_frontier_search_for_tsp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


def benchmark_on_tsp(number_of_locations, num_runs_per_algorithm, max_time_per_algorithm=10):
    """
    :param number_of_locations: number of randomly sampled locations
    :param num_runs_per_algorithm: number of TSP instances solved (the same instances are solved by all solvers)
    :param max_time_per_algorithm: time limit in seconds to be spent by the algorithms (might be exceeded by one step)
    :return: the fig, ax plot pair that illustrates the behavior of all three algorithms
    """

    histories_by_strategy = {}
    for strategy in ["dfs", "bfs", "ga"]:
        histories = []
        for seed in range(num_runs_per_algorithm):
            print(f"Running {strategy} with seed {seed}")
            random_state = np.random.RandomState(seed)
            tsp = TSP(n=number_of_locations, width_x=10, width_y=2, random_state=random_state)

            if strategy == "ga":
                search = get_genetic_search_for_tsp(tsp, random_state=np.random.RandomState(0), population_size=20)
            else:
                search = get_frontier_search_for_tsp(tsp=tsp, random_state=random_state, order=strategy)
            search.reset()
            t_start = time.time()
            times = []
            sols = []
            scores = []
            t = 0
            while search.active and (not times or times[-1] < max_time_per_algorithm):
                new_best = search.step()
                t += 1
                elapsed_time = np.round(time.time() - t_start, 2)
                if elapsed_time in times and new_best:
                    sols[-1] = search.best
                    scores[-1] = tsp.get_cost_of_route(search.best)
                elif elapsed_time not in times:
                    times.append(elapsed_time)
                    sols.append(search.best)
                    scores.append(tsp.get_cost_of_route(search.best) if new_best or ((len(scores) == 0 or np.isnan(scores[-1])) and search.best) else (scores[-1] if search.best else np.nan))
                assert not search.best or not np.isnan(scores[-1]), f"{strategy}"

            if search.best is not None:
                assert scores[-1] == tsp.get_cost_of_route(search.best)
                print(f"Best score: {scores[-1]} after a total runtime of {times[-1]}s for solution {search.best}")

                histories.append(pd.DataFrame({
                    "time": times,
                    "solution": sols,
                    "score": scores
                }))
        histories_by_strategy[strategy] = histories

    fig, ax = plt.subplots()

    for s_index, (strategy, histories) in enumerate(histories_by_strategy.items()):

        # fill up all time steps
        timesteps = set()
        for history in histories:
            timesteps |= set(history["time"])
        for i, history in enumerate(tqdm(histories)):
            missing_timesteps = [t for t in timesteps if t not in list(history["time"])]
            fill_hist = pd.DataFrame({"time": missing_timesteps, "solution": len(missing_timesteps) * [np.nan], "score": len(missing_timesteps) * [np.nan]})
            histories[i] = pd.concat([history, fill_hist], axis=0).sort_values("time")
            histories[i].ffill(axis=0, inplace=True)

        if histories:
            score_history_matrix = np.array([h["score"] for h in histories])
            time_axis = sorted(timesteps)
            if len(time_axis) <= 1:
                ax.scatter(time_axis, np.mean(score_history_matrix, axis=0), label=strategy, color=f"C{s_index}")
            else:
                ax.plot(time_axis, np.mean(score_history_matrix, axis=0), label=strategy, color=f"C{s_index}")
            ax.fill_between(time_axis, np.min(score_history_matrix, axis=0), np.max(score_history_matrix, axis=0), alpha=0.2, color=f"C{s_index}")
    ax.legend()
    return fig, ax
