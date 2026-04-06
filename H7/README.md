# Sheet 7 — Policy Evaluation & Value-Free Policy Construction

## Quick Reference: The Lake MDP

- Grid of cells: **S** (start), **F** (free), **H** (hole, terminal), **G** (goal, terminal)
- 4 actions: UP, RIGHT, DOWN, LEFT
- **Stochastic movement**: 80% intended direction, 10% each lateral direction
- **Rewards on entry**: F = -0.35, H = -1.0, G = +1.0, absorbing = 0
- After entering H or G, agent goes to absorbing state (game over)
- Discount factor: gamma in (0, 1]

---

## Core Concepts

### What is a Policy?

A policy pi(s) is a rule that tells the agent which action to take in each state. A **deterministic** policy maps each state to exactly one action. The central question: **how good is a given policy?**

### State-Value Function v^pi(s)

The value of state s under policy pi is the **expected total discounted reward** starting from s and following pi forever:

```
v^pi(s) = E[R_1 + gamma*R_2 + gamma^2*R_3 + ... | S_0 = s, policy = pi]
```

- gamma close to 1 = agent is patient (cares about future)
- gamma close to 0 = agent is myopic (only cares about now)

### The Bellman Equation (the key identity)

```
v(s) = r(s) + gamma * sum_s' P(s'|s, pi(s)) * v(s')
```

In words: **value of a state = expected immediate reward + discounted expected value of the next state**.

In matrix form with all states at once:

```
v = r + gamma * P * v
```

Where:
- `v` is a vector of values, one per state (length S)
- `r` is a vector of expected immediate rewards (length S)
- `P` is the S x S transition matrix: P[i,j] = probability of going from state i to state j under the policy
- `gamma` is the discount factor

---

## Exercise 1 — Bellman Update (bellman.py)

**One iteration** of the Bellman equation. Given a current guess `v`, produce a better guess:

```
v_new = r + gamma * P @ v
```

This is just plugging the current guess into the right side of the Bellman equation. Each application brings `v` closer to the true value. It's a **contraction**: the error shrinks by a factor of gamma each step.

### Why is it a contraction?

```
||B(v1) - B(v2)||_inf = ||gamma * P * (v1 - v2)||_inf
                      <= gamma * ||P|| * ||v1 - v2||_inf
                      <= gamma * ||v1 - v2||_inf      (since P is row-stochastic, ||P||_inf = 1)
```

Since gamma < 1, the operator shrinks distances. By the **Banach Fixed Point Theorem**, there exists a unique fixed point v^pi, and repeated application converges to it.

---

## Exercise 2 — Exact Policy Evaluation (bellman.py)

At the fixed point, `v = r + gamma * P * v`. Rearrange:

```
v - gamma * P * v = r
(I - gamma * P) * v = r
v = (I - gamma * P)^(-1) * r
```

We solve this linear system using `np.linalg.solve(A, r)` where `A = I - gamma * P`.

**Pros**: Gives the exact answer in one shot.
**Cons**: Requires solving an S x S linear system — O(S^3) time and O(S^2) memory. Impractical when S is very large (e.g., millions of states).

---

## Exercise 3 — Iterative Policy Evaluation (policy_eval.py)

Same answer as Exercise 2, but found by **looping**:

```
v <- 0          (initial guess)
repeat:
    v_new <- r + gamma * P @ v
    if ||v_new - v||_inf < eps * (1 - gamma) / gamma:
        stop
    v <- v_new
```

### The stopping criterion explained

The contraction mapping theorem gives us an error bound. If the change between iterations is small, the true error is also small:

```
||v_k - v^pi||_inf  <=  (gamma / (1 - gamma)) * ||v_k - v_{k-1}||_inf
```

So if `||v_new - v||_inf < eps * (1-gamma)/gamma`, then `||v_new - v^pi||_inf < eps`.

**Pros**: Only needs O(S) memory per iteration (just vectors, no big matrix inverse). Each iteration is O(S^2).
**Cons**: Needs many iterations to converge, especially when gamma is close to 1.

### Exact vs. Iterative — when to use which?

| | Exact | Iterative |
|---|---|---|
| Time | O(S^3) once | O(S^2) per iteration, many iterations |
| Memory | O(S^2) for the matrix | O(S) for the vectors |
| Best for | Small S (< ~10,000) | Large S or sparse P |
| Precision | Machine precision | Controlled by eps |

---

## Exercise 4 — Value-Free Policy Construction (my_policy.py)

**Critical constraint**: you must NOT use values, returns, or rewards to build the policy. Only topology/graph structure.

### The approach: BFS shortest path to goal

1. **Enumerate all states** reachable from S via transitions
2. **Build a deterministic graph** using the "most-likely successor" for each (state, action) pair — i.e., where the 80% probability takes you
3. **Reverse BFS from all Goal states** — compute d(s) = minimum number of deterministic steps to reach G. Holes are NOT seeded, so d(H) = infinity (holes are avoided naturally)
4. **Choose action greedily** — for each state, pick the action whose most-likely-successor has the smallest d. Tie-break with a fixed order (RIGHT > DOWN > LEFT > UP)

### Why it works

- BFS distance creates a "potential field" flowing toward the goal
- Holes have infinite distance, so the policy steers away from them
- The most-likely-successor simplification strips away stochasticity and treats the environment as deterministic for planning purposes
- This is purely structural — no reward values or discounted returns used

### Why can't we use values?

The exercise separates **policy construction** from **policy evaluation** to test your understanding of each independently. In a full algorithm (like **policy iteration**), you'd alternate: evaluate a policy, improve it using the values, evaluate again, etc. Here you only construct once and evaluate once.

---

## Exercise 5 — Runner (run.py)

Ties everything together in sequence:

```
1. pi = MyPolicy(mdp)                    -- build policy (value-free, Ex 4)
2. P, r = build_policy_Pr(mdp, pi)       -- extract transition matrix + reward vector
3. v = exact_policy_evaluation(P, r, gamma)  -- evaluate the policy (Ex 2 or 3)
4. fitness = v[start_state]              -- how good is this policy?
5. return (pi, v, fitness)
```

The **fitness** is v^pi(s_0): the expected total discounted reward the agent collects starting from S and following the constructed policy. Higher is better.

---

## Key Formulas Cheat Sheet

| Concept | Formula |
|---|---|
| Bellman equation | `v(s) = r(s) + gamma * sum P(s'|s,a) * v(s')` |
| Bellman update | `v_new = r + gamma * P @ v` |
| Exact evaluation | `v = solve(I - gamma*P, r)` |
| Iterative stop condition | `\|\|v_new - v\|\|_inf < eps * (1-gamma) / gamma` |
| Error bound after k steps | `\|\|v_k - v^pi\|\|_inf <= gamma^k / (1-gamma) * \|\|v_1 - v_0\|\|_inf` |
| Contraction factor | `\|\|B(v1) - B(v2)\|\|_inf <= gamma * \|\|v1 - v2\|\|_inf` |

---

## Glossary

- **MDP**: Markov Decision Process — a model of sequential decision-making with states, actions, transitions, and rewards
- **Policy (pi)**: a mapping from states to actions
- **Value function (v^pi)**: expected cumulative discounted reward under policy pi
- **Bellman operator (B)**: the mapping `v -> r + gamma * P * v`; applying it improves value estimates
- **Contraction**: a function that brings points closer together; guarantees convergence to a unique fixed point
- **Discount factor (gamma)**: how much future rewards are worth compared to immediate ones
- **Row-stochastic matrix**: each row sums to 1 (represents a probability distribution over next states)
- **Absorbing state**: a terminal state that loops to itself with reward 0 forever
- **Sup-norm / infinity norm**: max absolute value across all entries: `||v||_inf = max_i |v_i|`
