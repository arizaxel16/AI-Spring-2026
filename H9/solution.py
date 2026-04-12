import numpy as np
from policy import Policy
from lake_mdp import LakeMDP, ABSORB
from plot_utils import plot_policy
import matplotlib.pyplot as plt


class TabularPolicy(Policy):
    def __init__(self, mdp, rng, table=None):
        """
        A tabular deterministic policy represented as a dictionary mapping states to actions.

        Parameters
        ----------
        mdp : MDP
            The Markov Decision Process instance.
        rng : np.random.Generator
            Random number generator used for initialization.
        table : dict, optional
            A mapping from state to action to initialize the policy. If None, random actions are chosen.
        """
        super().__init__(mdp, rng)
        all_states = [
            ((i, j), cell)
            for i, row in enumerate(mdp.grid)
            for j, cell in enumerate(row)
        ]
        all_states.append((ABSORB, ABSORB))

        self.table = dict(table) if table is not None else {}
        for s in all_states:
            if s not in self.table:
                self.table[s] = rng.choice(list(mdp.actions(s)))

    def _decision(self, s):
        """
        Returns the action taken by the policy in state s.

        Parameters
        ----------
        s : State
            The current state.

        Returns
        -------
        Action
            The action chosen in state s.
        """
        return self.table[s]


def q_from_v(mdp, v, gamma):
    """
    Compute the Q-values from state-value estimates.

    Parameters
    ----------
    mdp : MDP
        The MDP for which Q-values are computed.
    v : dict
        State-value function mapping states to values.
    gamma : float
        Discount factor.

    Returns
    -------
    dict
        A nested dictionary q[s][a] representing the Q-values.
    """
    # TODO


def make_greedy_policy(mdp, v, gamma):
    """
    Generate a greedy policy based on state-value estimates.

    Parameters
    ----------
    mdp : MDP
        The MDP on which the policy is based.
    v : dict
        State-value function mapping states to values.
    gamma : float
        Discount factor.

    Returns
    -------
    dict
        A mapping from states to greedy actions.
    """
    # TODO


def policy_mismatch(q_true, q_est):
    """
    Identify states where greedy actions under two Q-functions differ.

    Parameters
    ----------
    q_true : dict
        The true Q-value function.
    q_est : dict
        The estimated Q-value function.

    Returns
    -------
    list
        States where the greedy actions differ.
    """
    # TODO


class ValueIteration:
    def __init__(self, gamma=0.99, epsilon=1e-3):
        """
        Initialize the Value Iteration algorithm.

        Parameters
        ----------
        gamma : float, optional
            Discount factor.
        epsilon : float, optional
            Convergence threshold (scaled by (1-gamma)/gamma).
        """
        self.gamma = gamma
        self.epsilon = epsilon

    def run(self, mdp):
        """
        Perform value iteration on the given MDP.

        Parameters
        ----------
        mdp : MDP
            The Markov Decision Process.

        Returns
        -------
        TabularPolicy
            The optimal policy derived from the converged value function.
        """
        # TODO


class PolicyEvaluationFactory:
    def __init__(
        self,
        mdp,
        gamma,
        policy,
        async_mode=False,
        subset=None,
        initial_values=None,
    ):
        """
        Factory for performing policy evaluation using either synchronous or asynchronous updates.

        Parameters
        ----------
        mdp : MDP
            The MDP being evaluated.
        gamma : float
            Discount factor.
        policy : Policy
            The policy to evaluate.
        async_mode : bool, optional
            If True, use asynchronous updates. Otherwise, use synchronous updates.
        subset : list, optional
            Subset of states to update (used in async mode).
        initial_values : dict, optional
            Initial value estimates.
        """
        self.mdp = mdp
        self.gamma = gamma
        self.policy = policy
        self.v = {}
        self.async_mode = async_mode
        self.subset = subset

        # TODO: Initialize state-values

    def synchronous_update(self):
        """
        Perform a synchronous update of the value function.

        Updates all states simultaneously using the Bellman expectation equation.
        """
        # TODO: Update self.v, that attribute will be used in the tests

    def asynchronous_update(self):
        """
        Perform an asynchronous update of the value function.

        Only updates the specified subset of states using the Bellman expectation equation.
        """
        # TODO: Update self.v, that attribute will be used in the tests

    def step(self):
        """
        Perform one step of policy evaluation.

        Uses synchronous or asynchronous update depending on initialization.
        """
        # TODO


class GeneralPolicyIteration:
    def __init__(
        self, mdp, gamma=0.99, steps_per_eval=5, async_mode=False, subset=None
    ):
        """
        Implements General Policy Iteration (GPI) combining policy evaluation and improvement.

        Parameters
        ----------
        mdp : MDP
            The Markov Decision Process.
        gamma : float, optional
            Discount factor.
        steps_per_eval : int, optional
            Number of evaluation steps per improvement iteration.
        async_mode : bool, optional
            Whether to use asynchronous evaluation.
        subset : list, optional
            Subset of states to evaluate in async mode.
        """
        self.mdp = mdp
        self.gamma = gamma
        self.steps_per_eval = steps_per_eval
        self.async_mode = async_mode
        self.subset = subset

    def run(self):
        """
        Run the GPI algorithm starting from an initial policy.

        Returns
        -------
        TabularPolicy
            The final improved policy after convergence.
        """
        # TODO


# Example usage and testing
if __name__ == "__main__":
    # Create Lake MDP
    grid = [
        ["S", "F", "F", "F"],
        ["F", "H", "F", "H"],
        ["F", "F", "F", "H"],
        ["H", "F", "F", "G"],
    ]
    mdp = LakeMDP(grid)
    # General Policy Iteration
    gpi = GeneralPolicyIteration(mdp, gamma=0.9, steps_per_eval=10, async_mode=False)
    gpi_policy = gpi.run()
    plot_policy(gpi_policy)
    # Value Iteration
    vi = ValueIteration(gamma=0.9)
    vi_policy = vi.run(mdp)
    plot_policy(vi_policy)
    # Show plots
    plt.show()
