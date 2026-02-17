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
    raise NotImplementedError


def crossover(random_state, p1, p2):
    """
    Takes two individuals and combines them into two new routes.

    Args:
        random_state (np.random.RandomState): random state to control random behavior
        p1 (list): parent tour 1 (location indices)
        p2 (list): parent tour 2 (location indices)

    Returns:
        list: A list of size 2 with the offsprings of the parents p1 and p2 as entries, which are also lists themselves
    """
    raise NotImplementedError


def mutate(random_state, i):
    """


    Args:
        random_state (np.random.RandomState): random state to control random behavior
        i (list): tour to be mutated

    Returns:
        list: a mutant copy of the given individual `i`
    """
    raise NotImplementedError


def run_genetic_search_for_gcs(
    gcs, timeout, random_state=np.random.RandomState(0), population_size=10
):
    """

    Tries for at most `timeout` seconds to find a good solution for the given GCS instance.

    Args:
        gcs: Search problem instance with `locations`, `is_better_route_than`, and `get_cost_of_route`.
        timeout (int): Timeout in seconds after which a solution must have been returned (ideally earlier).
        random_state (np.random.RandomState): random state to control random behavior.
        population_size (int): number of individuals kept in each generation.

    Returns:
        list: The best route found, represented as location indices.
    """
    raise NotImplementedError
