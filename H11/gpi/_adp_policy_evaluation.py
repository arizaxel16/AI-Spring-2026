from mdp._trial_interface import TrialInterface
import numpy as np

from policy_evaluation._linear import LinearSystemEvaluator
from gpi._trial_based_policy_evaluator import TrialBasedPolicyEvaluator
from mdp._base import ClosedFormMDP


def _has_action(action):
    return action is not None and not (
        isinstance(action, (float, np.floating)) and np.isnan(action)
    )


class ADPPolicyEvaluation(TrialBasedPolicyEvaluator):

    def __init__(
        self,
        trial_interface: TrialInterface,
        gamma: float,
        exploring_starts: bool,
        max_trial_length: int = np.inf,
        random_state: np.random.RandomState = None,
        precision_for_transition_probability_estimates=4,
        update_interval: int = 10,
    ):
        super().__init__(
            trial_interface=trial_interface,
            gamma=gamma,
            exploring_starts=exploring_starts,
            max_trial_length=max_trial_length,
            random_state=random_state,
        )
        self.precision = precision_for_transition_probability_estimates
        self.update_interval = update_interval

        self._known_states = []
        self._known_states_set = set()
        self._terminal_states = set()
        self._actions_in_state = {}
        self._rewards = {}

        # transition counts: _counts[s][a][s'] -> int; action counts: _action_counts[s][a]
        self._counts = {}
        self._action_counts = {}

        self._step_count = 0

    def _synchronize_knowledge_about_states_and_actions(self, s, a, r):
        if s not in self._known_states_set:
            self._known_states_set.add(s)
            self._known_states.append(s)
        self._rewards[s] = r
        if not _has_action(a):
            self._terminal_states.add(s)
        else:
            if s not in self._actions_in_state:
                self._actions_in_state[s] = set()
            self._actions_in_state[s].add(a)

    def get_believed_probs(self) -> dict:
        """
        :return: nested dict P[s][a][s'] = estimated P(s'|s,a) based on counts.
        """
        probs = {}
        for s, a_dict in self._counts.items():
            probs[s] = {}
            for a, succ_counts in a_dict.items():
                total = self._action_counts[s][a]
                if total == 0:
                    continue
                probs[s][a] = {
                    sp: round(c / total, self.precision)
                    for sp, c in succ_counts.items()
                }
        return probs

    def process_trial_for_policy(self, df_trial, policy):
        """
        :param df_trial: dataframe with the trial (three columns with states, actions, and the rewards)
        :param policy: the policy that was used to create the trial
        :return: a dictionary with a report of the step
        """
        states = list(df_trial.iloc[:, 0])
        actions = list(df_trial.iloc[:, 1])
        rewards = list(df_trial.iloc[:, 2])
        n = len(states)

        for t in range(n):
            self._synchronize_knowledge_about_states_and_actions(
                states[t], actions[t], rewards[t]
            )

        for t in range(n - 1):
            s, a = states[t], actions[t]
            if not _has_action(a):
                continue
            sp = states[t + 1]
            if s not in self._counts:
                self._counts[s] = {}
                self._action_counts[s] = {}
            if a not in self._counts[s]:
                self._counts[s][a] = {}
                self._action_counts[s][a] = 0
            self._counts[s][a][sp] = self._counts[s][a].get(sp, 0) + 1
            self._action_counts[s][a] += 1

        self._step_count += 1

        if self._step_count == 1 or (self._step_count % self.update_interval == 0):
            self._recompute_v_and_q(policy)

        return {
            "trial_length": n,
            "states_known": len(self._known_states),
        }

    def _recompute_v_and_q(self, policy):
        states = list(self._known_states)
        size = len(states)
        if size == 0:
            return
        idx = {s: i for i, s in enumerate(states)}

        P_pi = np.zeros((size, size))
        R = np.zeros(size)

        for i, s in enumerate(states):
            R[i] = self._rewards.get(s, 0.0)
            if s in self._terminal_states:
                continue
            a = policy(s)
            if (
                s in self._counts
                and a in self._counts[s]
                and self._action_counts[s][a] > 0
            ):
                total = self._action_counts[s][a]
                for sp, c in self._counts[s][a].items():
                    if sp in idx:
                        P_pi[i, idx[sp]] = c / total
            else:
                # action under policy not yet observed - treat as self-loop fallback
                P_pi[i, i] = 1.0

        try:
            B = np.eye(size) - self.gamma * P_pi
            v_vec = np.linalg.solve(B, R)
        except np.linalg.LinAlgError:
            gamma = min(1 - 1e-10, self.gamma)
            B = np.eye(size) - gamma * P_pi
            v_vec = np.linalg.solve(B, R)

        v = {s: float(v_vec[i]) for i, s in enumerate(states)}

        q = {}
        for s in states:
            if s in self._terminal_states:
                continue
            actions_seen = self._actions_in_state.get(s, set())
            if not actions_seen:
                continue
            q[s] = {}
            r_s = self._rewards.get(s, 0.0)
            for a in actions_seen:
                if (
                    s in self._counts
                    and a in self._counts[s]
                    and self._action_counts[s][a] > 0
                ):
                    total = self._action_counts[s][a]
                    expected = sum(
                        (c / total) * v.get(sp, 0.0)
                        for sp, c in self._counts[s][a].items()
                    )
                    q[s][a] = r_s + self.gamma * expected
                else:
                    q[s][a] = 0.0

        self.workspace.replace_v(v)
        self.workspace.replace_q(q)
