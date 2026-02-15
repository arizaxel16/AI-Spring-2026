import sys
sys.path.insert(0, "tsp")

import numpy as np
import time
from tsp import TSP
from gcs_ga_runner import (
    init, crossover, mutate,
    run_genetic_search_for_gcs,
)
from ga import GeneticSearch

# ---- Unit tests for operators ----
print("=== Operator tests ===")
rs = np.random.RandomState(42)
locs = list(range(10))

pop = init(locs, rs, 5)
assert len(pop) == 5
for ind in pop:
    assert sorted(ind) == locs
print("init: OK")

c1, c2 = crossover(rs, pop[0], pop[1])
assert sorted(c1) == locs
assert sorted(c2) == locs
print("crossover: OK")

m = mutate(rs, pop[0])
assert sorted(m) == locs
assert m != pop[0]
print("mutate: OK")

# ---- GeneticSearch class ----
print("\n=== GeneticSearch class ===")
tsp10 = TSP(n=10, random_state=np.random.RandomState(0))
rs2 = np.random.RandomState(0)

ga = GeneticSearch(
    init=lambda n: init(list(range(10)), rs2, n),
    crossover=lambda p1, p2: crossover(rs2, p1, p2),
    mutate=lambda i: mutate(rs2, i),
    better=tsp10.is_better_route_than,
    population_size=10,
)
ga.reset()
assert ga.best is not None
assert ga.active
assert len(ga.population) == 10

for gen in range(20):
    ga.step()
print(f"After 20 gens: cost = {tsp10.get_cost_of_route(ga.best):.4f}")
print("GeneticSearch: OK")

# ---- Full solver: 10 locations, 10s (1x1 area, matching grader) ----
print("\n=== 10 locations, 10s (1x1 area) ===")
tsp10 = TSP(n=10, random_state=np.random.RandomState(0))
t0 = time.time()
best = run_genetic_search_for_gcs(
    tsp10, timeout=10, random_state=np.random.RandomState(0), population_size=10
)
elapsed = time.time() - t0
cost = tsp10.get_cost_of_route(best)
print(f"  cost = {cost:.4f}  time = {elapsed:.2f}s")
print(f"  route = {best}")
if cost < 2.45:
    print("  PASSED (< 2.45 threshold)")
else:
    print(f"  WARNING: cost {cost:.4f} >= 2.45")

# ---- 10 locations, 10x2 area (matching benchmark) ----
print("\n=== 10 locations, 10s (10x2 area) ===")
tsp10b = TSP(n=10, random_state=np.random.RandomState(0), width_x=10, width_y=2)
t0 = time.time()
bestb = run_genetic_search_for_gcs(
    tsp10b, timeout=10, random_state=np.random.RandomState(0), population_size=10
)
elapsed = time.time() - t0
costb = tsp10b.get_cost_of_route(bestb)
print(f"  cost = {costb:.4f}  time = {elapsed:.2f}s")

# ---- 100 locations, 30s ----
print("\n=== 100 locations, 30s ===")
tsp100 = TSP(n=100, random_state=np.random.RandomState(0))
t0 = time.time()
best100 = run_genetic_search_for_gcs(
    tsp100, timeout=30, random_state=np.random.RandomState(0), population_size=10
)
elapsed = time.time() - t0
cost100 = tsp100.get_cost_of_route(best100)
print(f"  cost = {cost100:.4f}  time = {elapsed:.2f}s")
print("ALL DONE")
