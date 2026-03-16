from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Literal, Any, Type
import numpy as np

from mdp import MDP, Action
from policies import Policy

TerminalKind = Literal["goal", "hole", "none"]


@dataclass
class UtilityAnalyzer:
    mdp: MDP
    gamma: float = 0.99
    step_limit: int = 100

    def run_trial(
        self, policy_cls: Type[Policy], seed: int
    ) -> Tuple[float, int, TerminalKind]:
        """
        Instantiate a fresh policy with its own rng(seed) and simulate one episode.
        Returns (discounted_utility, length, terminal_kind).
        """
        rng = np.random.default_rng(seed)
        policy = policy_cls(self.mdp, rng)

        s = self.mdp.start_state()
        # Initial reward for entering the start state at t=0
        total_utility = self.mdp.reward(s)
        gamma_t = 1.0
        steps = 0

        last_grid_state = s # To track if we were on 'G' or 'H' before ⊥

        while not self.mdp.is_terminal(s) and steps < self.step_limit:
            last_grid_state = s
            a = policy(s)
            s, r = self.mdp.step(s, a, rng)

            steps += 1
            gamma_t *= self.gamma
            total_utility += gamma_t * r

        # Determine how the trial ended
        if not self.mdp.is_terminal(s):
            kind = "none" # Hit step limit
        else:
            # Check the character of the cell that triggered the move to ⊥
            r, c = last_grid_state
            cell = self.mdp.grid[r][c]
            if cell == 'G':
                kind = "goal"
            elif cell == 'H':
                kind = "hole"
            else:
                kind = "none"

        return total_utility, steps, kind

    def evaluate(
            self, policy_cls: Type[Policy], n_trials: int, base_seed: int = 0
    ) -> Dict[str, Any]:
        """Runs multiple trials and returns statistical summaries."""
        utilities = []
        lengths = []
        outcomes = {"goal": 0, "hole": 0, "none": 0}

        for i in range(n_trials):
            u, l, kind = self.run_trial(policy_cls, base_seed + i)
            utilities.append(u)
            lengths.append(l)
            outcomes[kind] += 1

        return {
            "n_trials": n_trials,
            "mean_utility": float(np.mean(utilities)),
            "utility_variance": float(np.var(utilities)),
            "p_goal": outcomes["goal"] / n_trials,
            "p_hole": outcomes["hole"] / n_trials,
            "p_none": outcomes["none"] / n_trials,
            "mean_length": float(np.mean(lengths))
        }