# template.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import numpy as np

from mdp import MDP, State, Action
from policy import Policy
from lake_mdp import UP, RIGHT, DOWN, LEFT, ABSORB
from mdp_utils import enumerate_states, build_policy_Pr

_ACTION_ORDER = (UP, RIGHT, DOWN, LEFT)


class TabularPolicy(Policy):
    """
    Deterministic or uniform-random tabular policy.

    A simple policy that, for each state, either:
      • returns the action stored in an internal table (deterministic), or
      • chooses uniformly at random among admissible actions if no table is provided.

    This class MUST:
      1) Be callable on a state s: `a = policy(s)` (implemented in the Policy base).
      2) Expose a probability distribution over actions via `probs(s)`:
         - If deterministic: 1.0 on the chosen action, 0.0 on others.
         - If random: uniform over admissible actions (no global RNG; use `self.rng`).
      3) Provide an alias `action_probs(s)` returning the same dict as `probs(s)`.
         The autograder will use this to build P^π and r^π without sampling.

    Parameters
    ----------
    mdp : MDP
        The environment implementing transition and reward interfaces.
    rng : np.random.Generator
        Random generator to use for all sampling in this policy.
    table : Optional[Dict[State, Action]]
        If provided, a deterministic mapping from state to action.
        If None, the policy acts as a uniform random policy.

    Notes
    -----
    • Terminal/absorbing states may have no admissible actions. In such cases,
      return ABSORB and a degenerate distribution {ABSORB: 1.0}.
    • Do not use global random state. Only use `self.rng`.
    """

    def __init__(
        self,
        mdp: MDP,
        rng: np.random.Generator,
        table: Optional[Dict[State, Action]] = None,
    ):
        raise NotImplementedError


def q_from_v(
    mdp: MDP, v: Dict[State, float], gamma: float
) -> Dict[Tuple[State, Action], float]:
    """
    Compute state–action values q^π(s, a) given state values v^π(s) for the same policy π.

    Definition
    ----------
    q^π(s,a) = Σ_{s'} P(s' | s, a) [ r(s') + γ v^π(s') ]

    Requirements
    ------------
    • Interpret rewards as “on-entry”: `mdp.reward(s_next)` is the reward for entering s'.
    • Use the MDP’s transition iterator: `for (s_next, p) in mdp.transition(s, a)`.
    • If a state has no admissible actions (terminal), define q(s, ABSORB) = 0.0.
    • Use `enumerate_states(mdp)` to iterate states in a consistent order.

    Parameters
    ----------
    mdp : MDP
        The environment with transitions and rewards.
    v : Dict[State, float]
        State-value function under the same policy π.
    gamma : float
        Discount factor γ ∈ (0, 1].

    Returns
    -------
    Dict[Tuple[State, Action], float]
        Mapping (s, a) ↦ q^π(s, a).

    Notes
    -----
    • The provided `v` is assumed to correspond to the same policy π implicitly
      used when `enumerate_states` and `transition` are evaluated.
    """
    raise NotImplementedError


def v_from_q(
    q: Dict[Tuple[State, Action], float], policy: TabularPolicy
) -> Dict[State, float]:
    """
    Compute state values v^π(s) given q^π(s, a) and a policy π.

    Deterministic π
    ---------------
    v^π(s) = q^π(s, π(s)).

    Stochastic (uniform) π
    ----------------------
    v^π(s) = Σ_a π(a|s) q^π(s, a).
    In this assignment, a stochastic TabularPolicy is uniform over admissible actions.

    Parameters
    ----------
    q : Dict[Tuple[State, Action], float]
        State–action values for a fixed policy π.
    policy : TabularPolicy
        A policy object providing action probabilities per state.

    Returns
    -------
    Dict[State, float]
        Mapping s ↦ v^π(s).

    Edge Cases
    ----------
    • Terminal states with no actions should return v(s) = 0.0.
    """
    raise NotImplementedError


def policy_evaluation(
    P: np.ndarray,
    r: np.ndarray,
    gamma: float,
    states: List[State],
    eps: float = 1e-6,
    max_iters: int = 100_000,
) -> Dict[State, float]:
    """
    Iterative policy evaluation (matrix form) returning v^π as a dict keyed by `states`.

    Fixed-point iteration
    ---------------------
    v_{k+1} = r + γ P v_k,
    stopping when ||v_{k+1} - v_k||_∞ < eps * (1-γ)/γ (for γ < 1).

    Exact solve (γ ≈ 1)
    -------------------
    When γ is numerically 1 (or extremely close), prefer the direct linear solve:
        (I - γ P) v = r

    Parameters
    ----------
    P : np.ndarray, shape (S, S)
        Row-stochastic transition matrix under the current policy π (built via `build_policy_Pr`).
    r : np.ndarray, shape (S,)
        Reward vector aligned with `states`. Reward convention: on entry to state.
    gamma : float
        Discount factor γ ∈ (0, 1].
    states : List[State]
        State ordering corresponding to rows/cols of P and entries of r.
    eps : float, default=1e-6
        Tolerance for the contraction-based stopping rule.
    max_iters : int, default=100000
        Hard cap on iterations (should not be hit for γ < 1 with a reasonable eps).

    Returns
    -------
    Dict[State, float]
        Mapping s ↦ v^π(s) using the same ordering as `states`.

    Notes
    -----
    • Use only NumPy (no SciPy). Use `np.linalg.solve` for the exact solve path.
    • For γ very close to 1, prefer the exact solve path to avoid slow convergence.
    """
    raise NotImplementedError


def _action_order_key(a: Action) -> Tuple[int, str]:
    """
    Stable tie-breaking key for actions.

    Returns a pair (rank, name) where:
      • rank is the index of `a` in `_ACTION_ORDER` (UP, RIGHT, DOWN, LEFT),
        or len(_ACTION_ORDER) if `a` is not present (e.g., ABSORB).
      • name is `str(a)` for lexicographic stability among unknown actions.

    This provides deterministic argmax behavior when q-values tie.

    Parameters
    ----------
    a : Action
        The action symbol to rank.

    Returns
    -------
    Tuple[int, str]
        Comparable key for sorted().
    """
    raise NotImplementedError


def policy_improvement(
    mdp: MDP, v: Dict[State, float], gamma: float
) -> Tuple[TabularPolicy, Dict[Tuple[State, Action], float]]:
    """
    Advantage-based policy improvement.

    Steps
    -----
    1) Compute q^π(s, a) from the provided v^π using `q_from_v`.
    2) Compute the advantage for every state–action pair:
           A^π(s, a) = q^π(s, a) - v^π(s).
    3) Improve the policy greedily:
           π'(s) = argmax_a A^π(s, a)
       with stable tie-breaking using `_ACTION_ORDER`.
    4) Return the improved deterministic TabularPolicy and the advantage dictionary.

    Parameters
    ----------
    mdp : MDP
        Environment used to enumerate states and actions.
    v : Dict[State, float]
        State-value function under the current policy π.
    gamma : float
        Discount factor γ ∈ (0, 1].

    Returns
    -------
    Tuple[TabularPolicy, Dict[Tuple[State, Action], float]]
        (π', advantage), where
         • π' is a deterministic `TabularPolicy` (has a `.table`),
         • advantage[(s, a)] = A^π(s, a) for all s and admissible a.

    Edge Cases
    ----------
    • If a state has no admissible actions, set π'(s) = ABSORB and skip arrows.
    """
    raise NotImplementedError


def policy_iteration(
    mdp: MDP,
    policy: Policy,
    gamma: float,
) -> Tuple[TabularPolicy, Dict[State, float]]:
    """
    Classic policy iteration (Howard's) until convergence.

    Loop
    ----
    repeat:
        1) Build P^π and r^π from current π via `build_policy_Pr(mdp, π, states)`.
        2) Evaluate π: v^π = policy_evaluation(P^π, r^π, γ, states).
        3) Improve: (π', A) = policy_improvement(mdp, v^π, γ).
    until π' == π  (stability check on the deterministic action table)

    Parameters
    ----------
    mdp : MDP
        The environment.
    policy : Policy
        Initial policy. May be a TabularPolicy with or without a table.
        If stochastic (no table), the first improvement will produce a deterministic table.
    gamma : float
        Discount factor γ ∈ (0, 1].

    Returns
    -------
    Tuple[TabularPolicy, Dict[State, float]]
        (π*, v^{π*}) where π* is stable (optimal for finite MDPs with γ ∈ (0,1)).

    Notes
    -----
    • Use `enumerate_states(mdp)` once per iteration to fix an ordering.
    • The stability check must be deterministic: compare action tables state by state.
    • Do not plot or print inside this function (keep it pure for autograding).
    """
    raise NotImplementedError


def get_optimal_policy(
    mdp: MDP,
    gamma: float,
    rng: np.random.Generator,
) -> TabularPolicy:
    """
    Convenience runner:
      1) Construct a uniform-random TabularPolicy (no table).
      2) Run policy iteration to convergence.
      3) Return the optimal deterministic policy π*.

    Parameters
    ----------
    mdp : MDP
        The environment instance (e.g., LakeMDP).
    gamma : float
        Discount factor γ ∈ (0, 1].
    rng : np.random.Generator
        RNG to initialize the starting random policy.

    Returns
    -------
    TabularPolicy
        Optimal deterministic policy (has a `.table` covering all states).

    Notes
    -----
    • Do not change function name or signature (autograded).
    • Do not print or plot here.
    """
    raise NotImplementedError
