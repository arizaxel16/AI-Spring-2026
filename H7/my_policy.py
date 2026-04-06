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
        self.tie_break = tie_break
        self._action_map: Dict[State, Action] = {}
        self._build()

    # -- Policy API ----------------------------------------------------------
    def _decision(self, s: State) -> Action:
        return self._action_map.get(s, ABSORB)

    # -- Internals -----------------------------------------------------------
    def _most_likely_successor(self, s: State, a: Action) -> State:
        transitions = self.mdp.transition(s, a)
        return max(transitions, key=lambda x: x[1])[0]

    def _build(self) -> None:
        from collections import deque

        states = enumerate_states(self.mdp)

        # Identify goal states (distance 0 seeds)
        goals = [s for s in states if isinstance(s, tuple) and len(s) == 2 and s[1] == "G"]

        # Build reverse adjacency: successor -> {predecessor states}
        reverse: Dict[State, List[State]] = {}
        # Also store forward info: state -> [(action, successor)]
        forward: Dict[State, List[Tuple[Action, State]]] = {}

        for s in states:
            if self.mdp.is_terminal(s):
                continue
            actions = list(self.mdp.actions(s))
            if actions == [ABSORB]:
                continue
            fwd = []
            for a in actions:
                ns = self._most_likely_successor(s, a)
                fwd.append((a, ns))
                reverse.setdefault(ns, []).append(s)
            forward[s] = fwd

        # Reverse BFS from goals to compute discrete distance d(s)
        dist: Dict[State, int] = {}
        queue = deque()
        for g in goals:
            dist[g] = 0
            queue.append(g)

        while queue:
            t = queue.popleft()
            for s in reverse.get(t, []):
                if s not in dist:
                    dist[s] = dist[t] + 1
                    queue.append(s)

        # For each non-terminal state, pick action whose most-likely-successor
        # has the smallest distance to goal; tie-break by self.tie_break order.
        for s in states:
            if self.mdp.is_terminal(s):
                self._action_map[s] = ABSORB
                continue
            actions = list(self.mdp.actions(s))
            if actions == [ABSORB]:
                self._action_map[s] = ABSORB
                continue

            best_action = None
            best_dist = math.inf
            for a in self.tie_break:
                if a not in actions:
                    continue
                ns = self._most_likely_successor(s, a)
                d = dist.get(ns, math.inf)
                if d < best_dist:
                    best_dist = d
                    best_action = a

            self._action_map[s] = best_action if best_action is not None else actions[0]
