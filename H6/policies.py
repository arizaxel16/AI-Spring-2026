from __future__ import annotations

from policy import Policy
from lake_mdp import DOWN, RIGHT, Action, State


class RandomPolicy(Policy):
    """Uniform over legal actions."""

    def _decision(self, s: State) -> Action:
        possible_actions = list(self.mdp.actions(s))
        return self.rng.choice(possible_actions)


class CustomPolicy(Policy):
    """
    Simple deterministic rule that avoids an immediate hole:
      - Prefer DOWN if the cell below is NOT a hole.
      - Else prefer RIGHT if the cell to the right is NOT a hole.
      - Else pick the first legal action (covers absorbing ⊥ case, or being boxed in).
    Note: This checks intended cells only (not slip outcomes).
    """

    def _decision(self, s: State) -> Action:
        if s == "⊥":
            return "⊥"

        r, c = s

        # Rule 1: Prefer DOWN if the cell below is NOT a hole
        # (Row increases as you go down)
        if 0 <= r + 1 < self.mdp.rows and self.mdp.grid[r+1][c] != 'H':
            return DOWN

        # Rule 2: Prefer RIGHT if the cell to the right is NOT a hole
        # (Column increases as you go right)
        if 0 <= c + 1 < self.mdp.cols and self.mdp.grid[r][c+1] != 'H':
            return RIGHT

        # Rule 3: Default to the first legal action
        # This covers cases where both choices are holes or you are at the edge
        return list(self.mdp.actions(s))[0]