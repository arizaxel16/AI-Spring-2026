try:
    from ._base import GeneralPolicyIterationComponent
except ImportError:
    from _base import GeneralPolicyIterationComponent
from mdp._trial_interface import TrialInterface
import numpy as np
import pandas as pd
from abc import abstractmethod


class TrialBasedPolicyEvaluator(GeneralPolicyIterationComponent):

    def __init__(
        self,
        trial_interface: TrialInterface,
        gamma: float,
        exploring_starts: bool,
        max_trial_length: int = np.inf,
        random_state: np.random.RandomState = None,
    ):
        super().__init__()
        self.trial_interface = trial_interface
        self.gamma = gamma
        self.exploring_starts = exploring_starts
        self.max_trial_length = max_trial_length
        self.rng = random_state if random_state is not None else np.random.RandomState()

    def _is_terminal(self, s):
        ti = self.trial_interface
        if hasattr(ti, "mdp") and ti.mdp is not None:
            return ti.mdp.is_terminal_state(s)
        return ti.is_terminal_state(s)

    def _generate_trial(self, policy):
        ti = self.trial_interface
        rows = []

        if self.exploring_starts:
            s, r = ti.get_random_state()
            if self._is_terminal(s):
                rows.append([s, None, r])
                return pd.DataFrame(rows, columns=["state", "action", "reward"])
            actions = ti.get_actions_in_state(s)
            a = actions[self.rng.choice(range(len(actions)))]
            rows.append([s, a, r])
            s, r = ti.exec_action(s, a)
            steps = 1
        else:
            s, r = ti.draw_init_state()
            steps = 0

        while not self._is_terminal(s) and steps < self.max_trial_length:
            a = policy(s)
            rows.append([s, a, r])
            s, r = ti.exec_action(s, a)
            steps += 1
        rows.append([s, None, r])
        return pd.DataFrame(rows, columns=["state", "action", "reward"])

    def step(self):
        if self.workspace.policy is None:
            raise ValueError(
                "A policy must be set in the workspace before calling step()."
            )
        if self.workspace.v is None:
            self.workspace.replace_v({})
        if self.workspace.q is None:
            self.workspace.replace_q({})

        trial = self._generate_trial(self.workspace.policy)
        info = self.process_trial_for_policy(trial, self.workspace.policy)

        return {"trial_length": len(trial), "info": info}

    @abstractmethod
    def process_trial_for_policy(self, trial, policy):
        raise NotImplementedError
