# Sheet 8 — Policy Iteration: From Evaluation to Optimal Policies

## Quick Reference: The Lake MDP (same as H7)

- Grid: **S** (start), **F** (free), **H** (hole, terminal), **G** (goal, terminal)
- Actions: UP, RIGHT, DOWN, LEFT
- Stochastic: 80% intended, 10% each lateral
- Rewards on entry: S = 0, F = -0.1, H = -1.0, G = +1.0, absorbing = 0
- After entering H or G, agent goes to absorbing state (game over)

---

## The Big Idea: Policy Iteration

H7 asked "how good is a given policy?" (evaluation). H8 asks "how do we find the **best** policy?" The answer is **policy iteration** — a loop of two steps:

```
1. EVALUATE:  Given policy pi, compute how good it is  -> v^pi
2. IMPROVE:   Use v^pi to build a better policy         -> pi'
3. Repeat until pi' == pi  (no more improvement possible = optimal!)
```

Policy iteration is guaranteed to converge to the optimal policy in a finite number of steps because:
- There are finitely many deterministic policies (|A|^|S|)
- Each improvement step produces a policy that's strictly better (or the same if already optimal)
- A strictly improving sequence over a finite set must terminate

---

## New Concepts in H8

### Q-values: q^pi(s, a)

While v^pi(s) tells you "how good is state s?", q^pi(s, a) tells you "how good is taking action a in state s?"

```
q^pi(s, a) = SUM_{s'} P(s'|s, a) * [reward(s') + gamma * v^pi(s')]
```

In words: "If I take action `a` from state `s`, I transition to `s'` with probability P. I collect the entry reward `reward(s')`, then from `s'` onward I follow policy pi and get discounted future value `gamma * v(s')`."

**Key relationship:** `v^pi(s) = q^pi(s, pi(s))` for deterministic policies. The value of a state equals the q-value of the action the policy chooses there.

### Advantage: A^pi(s, a)

```
A^pi(s, a) = q^pi(s, a) - v^pi(s)
```

The advantage answers: "How much **better** (or worse) is action `a` compared to what the current policy pi does?"

- A > 0: action `a` is better than the current policy at state s
- A = 0: action `a` is exactly as good as the current policy (could be the same action)
- A < 0: action `a` is worse than the current policy

The advantage at the policy's own action is always 0: `A(s, pi(s)) = q(s, pi(s)) - v(s) = 0`.

---

## Exercise 1 — TabularPolicy (solution.py)

A simple policy with two modes:

**Deterministic mode** (table provided): lookup the action for each state in a dictionary.
```python
if self.table is not None and s in self.table:
    return self.table[s]
```

**Random mode** (no table): choose uniformly at random among admissible actions.
```python
return self.rng.choice(actions)  # using the provided RNG, never global state
```

The `probs(s)` / `action_probs(s)` method returns the probability distribution:
- Deterministic: `{chosen_action: 1.0}`
- Random: `{UP: 0.25, RIGHT: 0.25, DOWN: 0.25, LEFT: 0.25}`

**Why `action_probs` matters:** The `build_policy_Pr` utility uses it to build the transition matrix P without sampling. For a random policy, P averages over all actions (each weighted by 0.25). For a deterministic policy, P uses just the one chosen action.

---

## Exercise 2 — Value Functions (solution.py)

### policy_evaluation: v^pi via iterative Bellman updates

Same as H7 but returns a `Dict[State, float]` instead of a numpy array:

```python
v_new = r + gamma * P @ v    # Bellman update
stop when ||v_new - v||_inf < eps * (1-gamma)/gamma
```

For gamma very close to 1, uses exact linear solve `(I - gamma*P) v = r` to avoid slow convergence.

### q_from_v: state-action values from state values

```python
for each (state s, action a):
    q(s,a) = SUM over all s' reachable via (s,a):
        P(s'|s,a) * [reward(s') + gamma * v(s')]
```

This computes what would happen if you take action `a` (possibly different from pi(s)) and then follow pi from the next state onward.

### v_from_q: state values from state-action values

```python
v(s) = SUM_a  pi(a|s) * q(s, a)
```

For deterministic pi: `v(s) = q(s, pi(s))` (just pick the one action).
For stochastic (uniform): average q over all actions.

---

## Exercise 3 — Policy Improvement (solution.py)

The core insight of reinforcement learning: **if you can find a single action better than the current policy at any state, you can improve the policy.**

### Steps:

1. **Compute q-values** from v^pi using `q_from_v`
2. **Compute advantages** for every (state, action): `A(s, a) = q(s, a) - v(s)`
3. **Act greedily**: pick the action with the highest advantage at each state

```python
pi'(s) = argmax_a  A(s, a)
```

Ties are broken using a fixed action order (UP > RIGHT > DOWN > LEFT) for deterministic behavior.

### The Policy Improvement Theorem

**If** `q^pi(s, pi'(s)) >= v^pi(s)` for all states s, **then** `v^{pi'}(s) >= v^{pi}(s)` for all s.

In words: if the new policy picks actions at least as good as the old policy everywhere, then the new policy is at least as good overall. Since we pick the argmax, the inequality holds with equality only when pi is already optimal.

**Proof sketch:**
```
v^pi(s) <= q^pi(s, pi'(s))                  # by construction (argmax)
         = E[R + gamma * v^pi(s')]            # expand q
         <= E[R + gamma * q^pi(s', pi'(s'))]  # apply improvement at s' too
         <= E[R + gamma*R' + gamma^2 * ...]   # keep unrolling
         = v^{pi'}(s)                          # definition of v^{pi'}
```

---

## Exercise 4 — Policy Iteration (solution.py)

The full loop:

```
pi = random policy
repeat:
    1. P, r = build_transition_matrix(mdp, pi)     # matrix form of pi
    2. v = policy_evaluation(P, r, gamma)            # how good is pi?
    3. pi', advantages = policy_improvement(mdp, v)  # build a better policy
    4. if pi' == pi: return (pi, v)                  # converged! pi is optimal
    5. pi = pi'                                      # continue with improved policy
```

### Convergence guarantee:
- Each improvement is strictly better (or equal = done)
- Finite number of deterministic policies: at most 4^|S| for the lake (4 actions, |S| states)
- Therefore terminates in finitely many steps
- In practice, converges in very few iterations (often < 10)

### Convergence check:
Compare the action tables state by state. If every state maps to the same action in both the old and new policy, we've converged.

---

## Exercise 5 — get_optimal_policy (solution.py)

Simple wrapper:

```python
1. Create random TabularPolicy (no table -> uniform random)
2. Run policy_iteration -> get optimal pi*
3. Return pi*
```

The beauty: **it doesn't matter where you start.** Any initial policy (even random) converges to the same optimal policy. Policy iteration always finds the global optimum for finite MDPs.

---

## How Everything Connects

```
                    TabularPolicy (pi)
                          |
                    build_policy_Pr
                          |
                     P matrix, r vector
                          |
                    policy_evaluation
                          |
                     v^pi (state values)
                          |
                    q_from_v
                          |
                     q^pi (state-action values)
                          |
              advantage = q(s,a) - v(s)
                          |
                    argmax -> pi' (improved policy)
                          |
                    pi' == pi?  ----yes----> DONE (optimal!)
                          |
                         no
                          |
                    pi = pi', go back to top
```

---

## Key Formulas Cheat Sheet

| Concept | Formula |
|---|---|
| State value | `v^pi(s) = SUM P(s'\|s,pi(s)) [r(s') + gamma*v(s')]` |
| Q-value | `q^pi(s,a) = SUM P(s'\|s,a) [r(s') + gamma*v(s')]` |
| v from q | `v^pi(s) = q^pi(s, pi(s))` |
| Advantage | `A^pi(s,a) = q^pi(s,a) - v^pi(s)` |
| Policy improvement | `pi'(s) = argmax_a A^pi(s,a)` |
| Improvement theorem | `A^pi(s,pi'(s)) >= 0 for all s => v^{pi'} >= v^pi` |
| Convergence | `pi* = pi iff A^pi(s,a) <= 0 for all s,a` |
| Max policies | `\|A\|^{\|S\|}` (e.g., 4^16 for 4x4 grid) |

---

## Comparison: H7 vs H8

| H7 | H8 |
|---|---|
| Evaluate a **given** policy | Find the **optimal** policy |
| Bellman update (1 step) | Full policy iteration (loop) |
| Only v^pi (state values) | v^pi AND q^pi (state-action values) |
| Value-free policy construction (BFS) | Value-based policy improvement (advantage) |
| No improvement step | Evaluate -> Improve -> Repeat |

H7 was about the **building blocks**. H8 uses those blocks to build the complete **policy iteration** algorithm.

---

## Glossary (new terms)

- **Q-value / State-action value q(s,a)**: expected return from taking action a in state s, then following policy pi
- **Advantage A(s,a)**: how much better action a is compared to the current policy: q(s,a) - v(s)
- **Policy improvement**: replacing the current policy with the greedy policy (argmax advantage)
- **Policy iteration**: alternating evaluation and improvement until convergence
- **Greedy policy**: always picks the action with highest q-value / advantage
- **Policy improvement theorem**: greedy improvement guarantees a better (or equal) policy
- **Optimal policy pi***: a policy where no improvement is possible; A*(s,a) <= 0 for all (s,a)
