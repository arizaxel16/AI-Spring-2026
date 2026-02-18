from general_constructive_search import GeneralConstructiveSearch
from collections import deque


def revise(bcn, X_i, X_j):
    """
    Returns a tuple (D_i', changed), where
        - D_i' is the maximal subset of the domain of X_i that is arc consistent with X_j
        - changed is a boolean value that is True if the domain is now smaller than before and False otherwise

    Args:
        bcn ((domains, constraints)): The BCN containing domains and binary constraints.
        X_i (Any): descriptor of the variable X_i
        X_j (Any): descriptor of the variable X_j
    """
    domains, constraints = bcn
    key = tuple(sorted([X_i, X_j]))
    constraint = constraints.get(key)
    if constraint is None:
        return set(domains[X_i]), False

    D_i = domains[X_i]
    D_j = domains[X_j]

    new_D_i = set()
    for v_i in D_i:
        for v_j in D_j:
            if constraint({X_i: v_i, X_j: v_j}):
                new_D_i.add(v_i)
                break

    return new_D_i, len(new_D_i) < len(D_i)


def ac3(bcn):
    """
    Reduces the domains in a BCN to make it arc consistent, if possible.

    Args:
        bcn ((domains, constraints)): The BCN to make arc consistent (if possible)

    Returns:
        (bcn', feasible), where
        - bcn' is the maximum subnetwork (in terms of domains) of bcn that is consistent
        - feasible is a boolean that is False if it could be verified that bcn doesn't have a solution and True otherwise
    """
    domains, constraints = bcn
    domains = {k: set(v) for k, v in domains.items()}

    neighbors = {var: set() for var in domains}
    for (A, B) in constraints:
        neighbors[A].add(B)
        neighbors[B].add(A)

    queue = deque()
    in_queue = set()
    for (A, B) in constraints:
        queue.append((A, B))
        queue.append((B, A))
        in_queue.add((A, B))
        in_queue.add((B, A))

    while queue:
        X_i, X_j = queue.popleft()
        in_queue.discard((X_i, X_j))

        D_i, changed = revise((domains, constraints), X_i, X_j)
        if changed:
            domains[X_i] = D_i
            if not D_i:
                return (domains, constraints), False
            for X_k in neighbors[X_i]:
                if X_k != X_j:
                    arc = (X_k, X_i)
                    if arc not in in_queue:
                        queue.append(arc)
                        in_queue.add(arc)

    return (domains, constraints), True


def get_general_constructive_search_for_bcn(bcn, phi=None):
    """
        Generates a GeneralConstructiveSearch that can find a solution in the search space described by the BCN.

    Args:
        bcn ((domains, constraints)): The BCN in which we look for a solution.
        phi (func, optional): Function that takes a dictionary of domains (variables are keys) and selects the variable to fix next.

    Returns:
        (search, decoder), where
         - search is a GeneralConstructiveSearch object
         - decoder is a function to decode a node to an assignment
    """
    domains, constraints = bcn

    # Build neighbor lists from constraints
    neighbors = {var: set() for var in domains}
    for (A, B) in constraints:
        neighbors[A].add(B)
        neighbors[B].add(A)

    # Apply AC-3 to get initial reduced domains
    reduced_bcn, feasible = ac3(bcn)
    if not feasible:
        search = GeneralConstructiveSearch(
            w0={}, succ=lambda n: [], goal=lambda n: False
        )
        search.OPEN = []
        return search, lambda n: n

    # Initial state: reduced domains as sets
    w0 = {var: set(vals) for var, vals in reduced_bcn[0].items()}

    def propagate(state, fixed_var):
        """Forward checking with unit propagation.
        When a variable becomes a singleton, remove inconsistent values
        from its neighbors' domains. Also checks singleton neighbors
        for conflicts."""
        queue = [fixed_var]
        while queue:
            var = queue.pop()
            val = next(iter(state[var]))
            for nb in neighbors[var]:
                key = tuple(sorted([var, nb]))
                constraint = constraints.get(key)
                if constraint is None:
                    continue
                if len(state[nb]) == 1:
                    # Check consistency with existing singleton
                    v_nb = next(iter(state[nb]))
                    if not constraint({var: val, nb: v_nb}):
                        return None
                    continue
                new_domain = set()
                for v_nb in state[nb]:
                    if constraint({var: val, nb: v_nb}):
                        new_domain.add(v_nb)
                if not new_domain:
                    return None
                if len(new_domain) < len(state[nb]):
                    state[nb] = new_domain
                    if len(new_domain) == 1:
                        queue.append(nb)
        return state

    def select_var(state):
        """MRV: select the unassigned variable with the smallest domain."""
        best = None
        best_size = float('inf')
        for var, domain in state.items():
            if len(domain) > 1:
                if len(domain) < best_size:
                    best_size = len(domain)
                    best = var
                    if best_size == 2:
                        return best
        return best

    def succ(state):
        if phi is not None:
            var = phi(state)
        else:
            var = select_var(state)
        if var is None:
            return []

        successors = []
        for val in sorted(state[var]):
            new_state = {v: set(d) for v, d in state.items()}
            new_state[var] = {val}
            new_state = propagate(new_state, var)
            if new_state is not None:
                successors.append(new_state)
        return successors

    def goal(state):
        return all(len(d) == 1 for d in state.values())

    def decoder(state):
        return {var: next(iter(d)) for var, d in state.items()}

    search = GeneralConstructiveSearch(w0=w0, succ=succ, goal=goal)
    return search, decoder


def get_binarized_constraints_for_all_diff(domains):
    """
        Derives all binary constraints that are necessary to make sure that all variables given in `domains` will have different values.

    Args:
        domains (dict): dictionary where keys are variable names and values are lists of possible values for the respective variable.

    Returns:
        dict: dictionary where keys are constraint names (it is recommended to use tuples, with entries in the tuple being the variable names sorted lexicographically) and values are the functions encoding the respective constraint set membership
    """
    constraints = {}
    variables = sorted(domains.keys())
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            v_i, v_j = variables[i], variables[j]
            constraints[(v_i, v_j)] = lambda assignment, a=v_i, b=v_j: assignment[a] != assignment[b]
    return constraints
