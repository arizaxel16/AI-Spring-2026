from __future__ import annotations
from typing import Dict, List, Tuple
import math
import numpy as np

from policy import Policy
from mdp import MDP, State, Action
from mdp_utils import enumerate_states

try:
    from lake_mdp import UP, RIGHT, DOWN, LEFT, ABSORB
except Exception:
    UP, RIGHT, DOWN, LEFT, ABSORB = "UP", "RIGHT", "DOWN", "LEFT", "⊥"


class MyPolicy(Policy):
    """
    Value-free constructive policy based on shortest-path over the MDP's
    most-likely-successor graph (no v^pi, no returns, no evaluation calls).

    Steps:
      1) Enumerate reachable states.
      2) Build a directed graph: for each (state, action), connect to the
         most likely successor under mdp.transition(s, a).
      3) Reverse-BFS from all goals to compute a discrete distance d(s).
      4) For each non-terminal state s, choose the action minimizing d(next).
         Tie-break with a fixed action order.

    Notes:
      • Holes are treated as terminals and are not seeded in BFS, so d(H)=∞,
        which naturally discourages stepping into holes unless unavoidable.
      • Absorbing and goals get d=0.
    """

    def __init__(
        self,
        mdp: MDP,
        rng: np.random.Generator,
        tie_break: Tuple[Action, ...] = (RIGHT, DOWN, LEFT, UP),
    ):
        super().__init__(mdp, rng)
        # TODO

    # -- Policy API ----------------------------------------------------------
    def _decision(self, s: State) -> Action:
        # TODO

    # -- Internals -----------------------------------------------------------
    def _most_likely_successor(self, s: State, a: Action) -> State:
        # TODO: This method is optional, but may help structure your code.

    def _build(self) -> None:
        # TODO
