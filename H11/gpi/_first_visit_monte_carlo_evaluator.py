import numpy as np

from mdp._trial_interface import TrialInterface
from gpi._trial_based_policy_evaluator import TrialBasedPolicyEvaluator


def _has_action(a):
    return a is not None and not (
        isinstance(a, (float, np.floating)) and np.isnan(a)
    )


class FirstVisitMonteCarloEvaluator(TrialBasedPolicyEvaluator):

    def __init__(
        self,
        trial_interface: TrialInterface,
        gamma: float,
        exploring_starts: bool,
        max_trial_length: int = np.inf,
        random_state: np.random.RandomState = None,
    ):
        super().__init__(
            trial_interface=trial_interface,
            gamma=gamma,
            exploring_starts=exploring_starts,
            max_trial_length=max_trial_length,
            random_state=random_state,
        )
        self._v_sum = {}
        self._v_count = {}
        self._q_sum = {}
        self._q_count = {}

    def process_trial_for_policy(self, df_trial, policy):
        """
        :param df_trial: dataframe with the trial (three columns with states, actions, and the rewards)
        :return: returns a depth-2 dictionary that contains the *change* in the q-values (np.inf if a q-value was not available before)
        """
        states = list(df_trial.iloc[:, 0])
        actions = list(df_trial.iloc[:, 1])
        rewards = list(df_trial.iloc[:, 2])
        n = len(states)

        if n == 0:
            return {}

        # Compute returns G[t] = r(S_t) + gamma * G[t+1]
        # (matches LinearSystemEvaluator convention where v(s) includes r(s))
        G = [0.0] * n
        G[n - 1] = float(rewards[n - 1])
        for t in range(n - 2, -1, -1):
            G[t] = float(rewards[t]) + self.gamma * G[t + 1]

        # First-visit times. The terminal row (last) has action=None; do not count it for q.
        first_visit_state = {}
        first_visit_sa = {}
        for t in range(n):
            s = states[t]
            a = actions[t]
            if s not in first_visit_state:
                first_visit_state[s] = t
            if _has_action(a) and (s, a) not in first_visit_sa:
                first_visit_sa[(s, a)] = t

        v = self.workspace.v
        q = self.workspace.q

        # Update state-value estimates for first visits
        for s, t in first_visit_state.items():
            self._v_count[s] = self._v_count.get(s, 0) + 1
            self._v_sum[s] = self._v_sum.get(s, 0.0) + G[t]
            v[s] = self._v_sum[s] / self._v_count[s]

        # Update q-value estimates for first visits of (s, a) and track changes
        changes = {}
        for (s, a), t in first_visit_sa.items():
            if s not in self._q_count:
                self._q_count[s] = {}
                self._q_sum[s] = {}
            had_value_before = a in self._q_count[s]
            old_q = q.get(s, {}).get(a, None)
            self._q_count[s][a] = self._q_count[s].get(a, 0) + 1
            self._q_sum[s][a] = self._q_sum[s].get(a, 0.0) + G[t]
            new_q = self._q_sum[s][a] / self._q_count[s][a]
            if s not in q:
                q[s] = {}
            q[s][a] = new_q
            if s not in changes:
                changes[s] = {}
            if had_value_before and old_q is not None:
                changes[s][a] = new_q - old_q
            else:
                changes[s][a] = np.inf

        return changes
