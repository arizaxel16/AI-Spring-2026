from ga import GeneticSearch
import numpy as np
import time


def init(locations, random_state, n):
    """
    Creates an initial random population of size n for the GCS problem.

    Args:
        locations (list): List of possible locations (just names/indices)
        random_state (np.random.RandomState): random state to control random behavior
        n (int): number of individuals in population

    Returns:
        list: a list of `n` individuals
    """
    pop = []
    indices = list(range(len(locations)))
    for _ in range(n):
        perm = list(indices)
        random_state.shuffle(perm)
        pop.append(perm)
    return pop


def crossover(random_state, p1, p2):
    """
    Order Crossover (OX): copies a segment from one parent, fills the rest
    from the other parent preserving relative order.  Guarantees two valid
    offspring routes.

    Args:
        random_state (np.random.RandomState): random state to control random behavior
        p1 (list): parent tour 1 (location indices)
        p2 (list): parent tour 2 (location indices)

    Returns:
        list: A list of size 2 with the offsprings of the parents p1 and p2 as entries
    """
    n = len(p1)
    if n <= 1:
        return [list(p1), list(p2)]

    c1, c2 = sorted(random_state.choice(n, 2, replace=False))
    # Ensure segment doesn't cover entire route (would produce trivial copies)
    if c2 - c1 >= n - 1:
        c2 -= 1

    def _make_child(donor, filler):
        child = [None] * n
        child[c1 : c2 + 1] = donor[c1 : c2 + 1]
        placed = set(child[c1 : c2 + 1])
        fill_values = [v for v in filler if v not in placed]
        fill_pos = list(range(c2 + 1, n)) + list(range(0, c1))
        for pos, val in zip(fill_pos, fill_values):
            child[pos] = val
        return child

    return [_make_child(p1, p2), _make_child(p2, p1)]


def mutate(random_state, i):
    """
    Inversion mutation: reverses a random sub-segment of the tour.
    Equivalent to a single 2-opt move.  Always produces a different route
    (segment length >= 2).

    Args:
        random_state (np.random.RandomState): random state to control random behavior
        i (list): tour to be mutated

    Returns:
        list: a mutant copy of the given individual `i`
    """
    route = list(i)
    n = len(route)
    if n <= 1:
        return route
    c1, c2 = sorted(random_state.choice(n, 2, replace=False))
    route[c1 : c2 + 1] = route[c1 : c2 + 1][::-1]
    return route


# ---------------------------------------------------------------------------
# Competition-grade helpers used by run_genetic_search_for_gcs
# ---------------------------------------------------------------------------

def _get_distance_matrix(gcs):
    """Get or build the distance matrix for fast delta evaluations."""
    if hasattr(gcs, "distances"):
        return gcs.distances
    n = len(gcs.locations)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            d = np.linalg.norm(
                np.array(gcs.locations[i]) - np.array(gcs.locations[j])
            )
            dist[i, j] = dist[j, i] = d
    return dist


def _two_opt(route, dist):
    """
    Fast 2-opt local search using O(1) delta evaluation.
    Repeatedly reverses sub-segments that shorten the path.
    """
    n = len(route)
    best = list(route)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Delta cost of reversing best[i:j+1]
                delta = 0.0
                if i > 0:
                    delta += dist[best[i - 1], best[j]] - dist[best[i - 1], best[i]]
                if j < n - 1:
                    delta += dist[best[i], best[j + 1]] - dist[best[j], best[j + 1]]
                if delta < -1e-10:
                    best[i : j + 1] = best[i : j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
    return best


def _nearest_neighbor(dist, n, start):
    """Greedy nearest-neighbour heuristic starting from *start*."""
    visited = set([start])
    route = [start]
    cur = start
    for _ in range(n - 1):
        best_j = -1
        best_d = float("inf")
        for j in range(n):
            if j not in visited and dist[cur, j] < best_d:
                best_d = dist[cur, j]
                best_j = j
        route.append(best_j)
        visited.add(best_j)
        cur = best_j
    return route


def _double_bridge(route, random_state):
    """
    Double-bridge perturbation: splits the route into four segments and
    reconnects them as (A, C, B, D).  This escapes 2-opt local optima while
    keeping the route valid.
    """
    n = len(route)
    if n < 4:
        r = list(route)
        if n >= 2:
            i, j = sorted(random_state.choice(n, 2, replace=False))
            r[i], r[j] = r[j], r[i]
        return r
    cuts = sorted(random_state.choice(range(1, n), 3, replace=False))
    a, b, c = cuts
    return route[:a] + route[c:] + route[b:c] + route[a:b]


def run_genetic_search_for_gcs(
    gcs, timeout, random_state=np.random.RandomState(0), population_size=10
):
    """
    Tries for at most `timeout` seconds to find a good solution for the given
    GCS instance.  Three-phase strategy:

      1. Seed high-quality routes via nearest-neighbour + 2-opt from every city.
      2. Iterated Local Search: double-bridge perturbation + 2-opt (main engine).
      3. GA bursts for diversification, refined with 2-opt.

    Args:
        gcs: Search problem instance with `locations`, `is_better_route_than`,
             and `get_cost_of_route`.
        timeout (int): Timeout in seconds.
        random_state (np.random.RandomState): random state to control random
            behavior.
        population_size (int): number of individuals kept in each generation.

    Returns:
        list: The best route found, represented as location indices.
    """
    n_locs = len(gcs.locations)
    locations = list(range(n_locs))
    dist = _get_distance_matrix(gcs)
    start_time = time.time()

    best_overall = None
    best_cost = float("inf")

    def _update_best(route):
        nonlocal best_overall, best_cost
        c = gcs.get_cost_of_route(route)
        if c < best_cost:
            best_cost = c
            best_overall = list(route)

    deadline = start_time + timeout - 0.5

    # --- Phase 1: nearest-neighbour + 2-opt from every starting city --------
    for s in range(n_locs):
        if time.time() >= min(deadline, start_time + timeout * 0.2):
            break
        route = _nearest_neighbor(dist, n_locs, s)
        route = _two_opt(route, dist)
        _update_best(route)

    # --- Phase 2: ILS main loop with random restarts -------------------------
    current = list(best_overall) if best_overall else locations
    current_cost = gcs.get_cost_of_route(current) if best_overall else float("inf")
    ils_stale = 0

    while time.time() < deadline:
        perturbed = _double_bridge(current, random_state)
        improved = _two_opt(perturbed, dist)
        imp_cost = gcs.get_cost_of_route(improved)

        if imp_cost < current_cost:
            current = improved
            current_cost = imp_cost
            _update_best(current)
            ils_stale = 0
        else:
            ils_stale += 1
            if ils_stale > 20:
                # Random restart: fresh shuffle + 2-opt
                perm = list(locations)
                random_state.shuffle(perm)
                current = _two_opt(perm, dist)
                current_cost = gcs.get_cost_of_route(current)
                _update_best(current)
                ils_stale = 0

    # --- Phase 3: one quick GA burst to diversify further -------------------
    if time.time() < start_time + timeout - 0.2:
        ga = GeneticSearch(
            init=lambda n: init(locations, random_state, n),
            crossover=lambda p1, p2: crossover(random_state, p1, p2),
            mutate=lambda i: mutate(random_state, i),
            better=gcs.is_better_route_than,
            population_size=population_size,
        )
        ga.reset()
        if best_overall is not None and ga.population:
            ga.population[0] = list(best_overall)
        for _ in range(50):
            if time.time() >= start_time + timeout - 0.1:
                break
            ga.step()
        if ga.best is not None:
            refined = _two_opt(ga.best, dist)
            _update_best(refined)

    # Safety: guarantee we always return something
    if best_overall is None:
        best_overall = locations

    return best_overall
