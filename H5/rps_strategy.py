# rps_strategy.py
from __future__ import annotations
import numpy as np
from strategy import Strategy


class RPSStrategy(Strategy):
    """
    Rock–Paper–Scissors strategy under known opponent distribution (risk).
    """

    PAYOFF_TABLE = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)

    def __init__(self, utility_fn, mode="expected_utility", origin=None, mapping=None):
        """
        Initialize the RPS strategy.

        Parameters
        ----------
        utility_fn : callable
            Function mapping a payoff matrix (ndarray) to a utility matrix.
        mode : {'expected_payoff', 'expected_utility'}, default='expected_utility'
            Criterion used to evaluate actions.
        origin : dict of {int: float}
            Base probability measure on Ω. Keys are sample points ω,
            values are probabilities.
        mapping : dict of {int: list}
            Random variable X. Keys are outcome categories in {0,1,2},
            values are lists of points w that map to that category.
        """
        pass  # TODO

    def decision(self, rng: np.random.Generator) -> int:
        """
        Choose the best action given the opponent distribution.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator for tie-breaking.

        Returns
        -------
        action : int
            The chosen action in {0=Rock, 1=Paper, 2=Scissors}.
        """
        pass  # TODO
