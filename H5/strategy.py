# strategy.py
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class Strategy(ABC):
    def compute_distribution(self, origin: dict, mapping: dict) -> np.ndarray:
        """
        Compute the induced distribution of a random variable X over {0,1,2}.

        Parameters
        ----------
        origin : dict of {int: float}
            Base probability measure on Ω. Keys are sample points ω,
            values are probabilities.
        mapping : dict of {int: list}
            Random variable X. Keys are outcome categories in {0,1,2},
            values are lists of points w that map to that category.


        Returns
        -------
        dist : ndarray of shape (3,)
            Normalized distribution (P[X=0], P[X=1], P[X=2]).
        """
        x_dist = {0: 0.0, 1: 0.0, 2: 0.0}

        total_mass = sum(origin.values())

        for x, omegas in mapping.items():
            for omega in omegas:
                x_dist[x] += origin.get(omega, 0)

        dist = np.array([x_dist[0], x_dist[1], x_dist[2]])

        if total_mass > 0:
            dist = dist / total_mass

        return dist

    def expected_payoff(self, opponent_dist, payoff_table) -> np.ndarray:
        """
        Compute expected payoff of each action.

        Parameters
        ----------
        opponent_dist : array-like of shape (3,) or dict {int: float}
            Opponent distribution over {0,1,2}. If dict, must have keys 0,1,2.
        payoff_table : array-like of shape (3,3)
            Payoff matrix: rows = our actions a, cols = opponent outcomes o.

        Returns
        -------
        values : ndarray of shape (3,)
            Expected payoff of each action a=0,1,2.
        """
        p_o = np.array(list(opponent_dist.values())) if isinstance(opponent_dist, dict) else np.array(opponent_dist)

        return payoff_table.dot(p_o)

    def expected_utility(self, opponent_dist, payoff_table, utility_fn) -> np.ndarray:
        """
        Compute expected utility of each action.

        Parameters
        ----------
        opponent_dist : array-like of shape (3,) or dict {int: float}
            Opponent distribution over {0,1,2}.
        payoff_table : array-like of shape (3,3)
            Payoff matrix.
        utility_fn : callable
            Function applied elementwise to the payoff_table.

        Returns
        -------
        values : ndarray of shape (3,)
            Expected utility of each action a=0,1,2.
        """
        utility_payoff_table = np.vectorize(utility_fn)(payoff_table)
        p_o = np.array(list(opponent_dist.values())) if isinstance(opponent_dist, dict) else np.array(opponent_dist)

        return utility_payoff_table.dot(p_o)

    @abstractmethod
    def decision(self, rng: np.random.Generator) -> int:
        """
        Decide which action to take.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator for tie-breaking.

        Returns
        -------
        action : int
            The chosen action in {0=Rock, 1=Paper, 2=Scissors}.
        """
        raise NotImplementedError("Subclasses must implement this method.")
