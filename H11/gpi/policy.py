from __future__ import annotations

import math
import time

import numpy as np

from connect4.policy import Policy
from connect4.trial_interface import Connect4TrialInterface


def learn_policy(
    trial_interface: Connect4TrialInterface,
    timeout: int | None = None,
) -> Policy:
    """
    Learn a policy against a fixed (seeded, deterministic) opponent.

    Strategy: since the opponent is deterministic given the board, exhaustive
    minimax-style search from the initial state finds an action that yields
    the highest reward (+1 for our win, 0 draw, -1 loss). The search is
    memoized by board contents.
    """
    deadline = time.time() + timeout if timeout is not None else None

    memo = {}

    def _key(state):
        return np.asarray(state, dtype=int).tobytes()

    def _time_left():
        if deadline is None:
            return True
        return time.time() < deadline

    def search(state):
        key = _key(state)
        if key in memo:
            return memo[key]

        if trial_interface.is_terminal_state(state):
            r = float(trial_interface.get_reward(state))
            memo[key] = (r, None)
            return r, None

        actions = trial_interface.get_actions_in_state(state)
        best_v = -np.inf
        best_a = actions[0]

        for a in actions:
            if not _time_left():
                break
            try:
                next_state, r = trial_interface.exec_action(state, a)
            except Exception:
                continue
            if trial_interface.is_terminal_state(next_state):
                v = float(r)
            else:
                v_sub, _ = search(next_state)
                v = v_sub
            if v > best_v:
                best_v = v
                best_a = a
                if best_v >= 1.0:
                    break

        memo[key] = (best_v, best_a)
        return best_v, best_a

    init_state, _ = trial_interface.draw_init_state()
    try:
        search(init_state)
    except RecursionError:
        pass

    action_table = {k: a for k, (_, a) in memo.items() if a is not None}

    class _LearnedPolicy(Policy):
        def __init__(self, table, ti):
            self._table = table
            self._ti = ti

        def act(self, s: np.ndarray) -> int:
            key = np.asarray(s, dtype=int).tobytes()
            if key in self._table:
                return int(self._table[key])
            actions = self._ti.get_actions_in_state(s)
            if not actions:
                return 0
            return int(actions[0])

    return _LearnedPolicy(action_table, trial_interface)
