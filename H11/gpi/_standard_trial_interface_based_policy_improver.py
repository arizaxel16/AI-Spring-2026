try:
    from ._base import GeneralPolicyIterationComponent
except ImportError:
    from _base import GeneralPolicyIterationComponent
from mdp._trial_interface import TrialInterface
import numpy as np


class StandardTrialInterfaceBasedPolicyImprover(GeneralPolicyIterationComponent):

    def __init__(self, trial_interface: TrialInterface, random_state: np.random.RandomState):
        super().__init__()
        self.trial_interface = trial_interface
        self.rng = random_state if random_state is not None else np.random.RandomState()
        # Cache the action set per state so the trial interface is queried only once per state
        self._actions_cache = {}

    def _get_actions(self, s):
        if s not in self._actions_cache:
            self._actions_cache[s] = list(self.trial_interface.get_actions_in_state(s))
        return self._actions_cache[s]

    def _greedy_policy(self, s):
        actions = self._get_actions(s)
        if not actions:
            return None
        q = self.workspace.q if self.workspace.q is not None else {}
        q_s = q.get(s, {})

        best_v = -np.inf
        best_actions = []
        for a in actions:
            # unknown q-values are initialized with 0
            v_a = q_s.get(a, 0.0)
            if v_a > best_v:
                best_v = v_a
                best_actions = [a]
            elif v_a == best_v:
                best_actions.append(a)

        if len(best_actions) == 1:
            return best_actions[0]
        return best_actions[self.rng.choice(range(len(best_actions)))]

    def step(self):
        # Greedy policy reads workspace.q at call time, so a single bound function suffices.
        if self.workspace.policy is not self._greedy_policy:
            self.workspace.replace_policy(self._greedy_policy)
        return {}
