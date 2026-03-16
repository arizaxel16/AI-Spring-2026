from __future__ import annotations
from typing import Iterable, List, Tuple, Dict

from mdp import MDP, State, Action

UP, RIGHT, DOWN, LEFT, ABSORB = "UP", "RIGHT", "DOWN", "LEFT", "⊥"


class LakeMDP(MDP):
    """
    Grid map (matrix of single-character strings), e.g.:
      [
        ['S','F','F','F'],
        ['F','H','F','F'],
        ['F','F','F','F'],
        ['H','F','F','G'],
      ]

    Rewards are *state-entry* rewards. After entering H or G, the next state is
    the absorbing state ⊥ with only legal action ⊥ and 0 reward forever.
    """

    def __init__(self, grid: Iterable[Iterable[str]]):
        # Convert the iterable into a list of lists for indexing
        self.grid = [list(row) for row in grid]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows > 0 else 0

        # Pre-calculate the starting coordinates
        self._start_coords = None
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 'S':
                    self._start_coords = (r, c)

    # --- MDP interface -----------------------------------------------------
    def start_state(self) -> State:
        return self._start_coords

    def actions(self, s: State) -> Iterable[Action]:
        # If we are in the synthetic absorbing state, we can only ABSORB
        if s == ABSORB:
            return [ABSORB]

        # If we just landed on a Hole or Goal, the next step must be to ABSORB
        r, c = s
        if self.grid[r][c] in ('H', 'G'):
            return [ABSORB]

        # Standard movement actions for S and F cells
        return [UP, RIGHT, DOWN, LEFT]

    def reward(self, s: State) -> float:
        if s == ABSORB:
            return 0.0

        r, c = s
        cell = self.grid[r][c]

        if cell in ('F', 'S'):
            return 0.1
        if cell == 'H':
            return -1.0
        if cell == 'G':
            return 1.0
        return 0.0

    def is_terminal(self, s: State) -> bool:
        # A trial only ends when we reach the synthetic state ⊥
        return s == ABSORB

    def transition(self, s: State, a: Action) -> List[Tuple[State, float]]:
        # 1. Handle transitions from already terminal/absorbing states
        if s == ABSORB:
            return [(ABSORB, 1.0)]

        r, c = s
        if self.grid[r][c] in ('H', 'G'):
            return [(ABSORB, 1.0)]

        # 2. Handle standard movement with 0.8 accuracy and 0.1 slip chance
        # Map lateral directions for each action
        slips = {
            UP: [LEFT, RIGHT],
            DOWN: [LEFT, RIGHT],
            LEFT: [UP, DOWN],
            RIGHT: [UP, DOWN]
        }

        results = [(self._get_dest(s, a), 0.8)]
        # Intended move (80%)

        # Lateral slips (10% each)
        for lateral_action in slips.get(a, []):
            results.append((self._get_dest(s, lateral_action), 0.1))

        return results

    # --- helpers -----------------------------------------------------------
    def _get_dest(self, s: Tuple[int, int], a: Action) -> State:
        """Calculates destination coordinates; stays in place if hitting a border."""
        r, c = s
        nr, nc = r, c

        if a == UP: nr -= 1
        elif a == DOWN: nr += 1
        elif a == LEFT: nc -= 1
        elif a == RIGHT: nc += 1

        # Boundary check: If the new coordinates are within the grid, return them
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            return (nr, nc)

        # Otherwise, the agent "bumps" and stays in the original square
        return (r, c)