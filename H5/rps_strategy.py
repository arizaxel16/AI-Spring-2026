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

        if origin is None or mapping is None:
            origin = {i: 0.1 for i in range(1, 11)}
            mapping = {
                0: [1, 2, 3, 4, 5], # Rock
                1: [6, 7, 8, 9],    # Paper
                2: [10]             # Scissors
            }

        self.utility_fn = utility_fn
        self.mode = mode

        self.opponent_dist = self.compute_distribution(origin, mapping)

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

        if self.mode == 'expected_payoff':
            values = self.expected_payoff(self.opponent_dist, self.PAYOFF_TABLE)
        else:
            values = self.expected_utility(self.opponent_dist, self.PAYOFF_TABLE, self.utility_fn)

        max_val = np.max(values)

        winners = np.where(np.isclose(values, max_val))[0]

        chosen_action = rng.choice(winners)

        return int(chosen_action)
