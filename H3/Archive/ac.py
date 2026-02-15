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
    swap = key[0] != X_i

    new_D_i = set()
    for v_i in D_i:
        for v_j in D_j:
            if (constraint(v_j, v_i) if swap else constraint(v_i, v_j)):
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


def _detect_sudoku_groups(domains):
    """Detect Sudoku row/col/block groups from (row, col) variable names."""
    vars_list = list(domains.keys())
    if not all(isinstance(v, tuple) and len(v) == 2 for v in vars_list):
        return None
    n_vars = len(vars_list)
    n = int(round(n_vars ** 0.5))
    if n * n != n_vars:
        return None
    k = int(round(n ** 0.5))
    if k * k != n:
        return None
    groups = []
    for r in range(n):
        groups.append(tuple((r, c) for c in range(n)))
    for c in range(n):
        groups.append(tuple((r, c) for r in range(n)))
    for bi in range(k):
        for bj in range(k):
            groups.append(tuple(
                (r, c)
                for r in range(bi * k, (bi + 1) * k)
                for c in range(bj * k, (bj + 1) * k)
            ))
    return groups


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

    # --- Precompute neighbour lists (once) ---
    _nbr_set = {var: set() for var in domains}
    for (A, B) in constraints:
        _nbr_set[A].add(B)
        _nbr_set[B].add(A)
    neighbors = {var: list(nbs) for var, nbs in _nbr_set.items()}

    # --- Detect Sudoku groups for advanced propagation ---
    groups = _detect_sudoku_groups(domains)

    # --- Apply AC-3 to get initial reduced domains ---
    reduced_bcn, feasible = ac3(bcn)
    if not feasible:
        search = GeneralConstructiveSearch(
            w0={}, succ=lambda n: [], goal=lambda n: False
        )
        search.OPEN = []
        return search, lambda n: n

    # --- Bitmask helpers ---
    def _to_bitmask(vals):
        m = 0
        for v in vals:
            m |= 1 << (v - 1)
        return m

    def _from_bitmask(m):
        vals = set()
        while m:
            lsb = m & -m
            vals.add(lsb.bit_length())
            m &= m - 1
        return vals

    _popcount = lambda x: bin(x).count('1')

    # --- Convert initial domains to bitmasks ---
    w0 = {var: _to_bitmask(vals) for var, vals in reduced_bcn[0].items()}

    # --- Fast singleton propagation (equivalent to AC-3 for all-diff) ---
    def _propagate(state, fixed_var):
        queue = [fixed_var]
        while queue:
            var = queue.pop()
            val_bit = state[var]
            for nb in neighbors[var]:
                nb_mask = state[nb]
                if nb_mask & val_bit:
                    new_mask = nb_mask & ~val_bit
                    if new_mask == 0:
                        return None
                    state[nb] = new_mask
                    if not (new_mask & (new_mask - 1)):
                        queue.append(nb)
        return state

    # --- Advanced propagation: hidden singles + naked pairs ---
    def _advanced(state):
        if not groups:
            return state
        progress = True
        while progress:
            progress = False
            for group in groups:
                unfixed = []
                fixed_bits = 0
                for var in group:
                    m = state[var]
                    if m & (m - 1):
                        unfixed.append(var)
                    else:
                        fixed_bits |= m
                if not unfixed:
                    continue

                # -- Hidden singles --
                all_unfixed = 0
                for var in unfixed:
                    all_unfixed |= state[var]
                needed = all_unfixed & ~fixed_bits
                remaining = needed
                while remaining:
                    bit = remaining & -remaining
                    remaining &= remaining - 1
                    cnt = 0
                    candidate = None
                    for var in unfixed:
                        if state[var] & bit:
                            cnt += 1
                            candidate = var
                            if cnt > 1:
                                break
                    if cnt == 0:
                        return None
                    if cnt == 1 and state[candidate] != bit:
                        state[candidate] = bit
                        state = _propagate(state, candidate)
                        if state is None:
                            return None
                        progress = True
                        break
                if progress:
                    break

                # -- Naked pairs --
                for i in range(len(unfixed)):
                    mi = state[unfixed[i]]
                    if _popcount(mi) != 2:
                        continue
                    for j in range(i + 1, len(unfixed)):
                        if state[unfixed[j]] == mi:
                            for k in range(len(unfixed)):
                                if k != i and k != j:
                                    old = state[unfixed[k]]
                                    nw = old & ~mi
                                    if nw != old:
                                        if nw == 0:
                                            return None
                                        state[unfixed[k]] = nw
                                        progress = True
                                        if not (nw & (nw - 1)):
                                            state = _propagate(state, unfixed[k])
                                            if state is None:
                                                return None
                            if progress:
                                break
                    if progress:
                        break
                if progress:
                    break
        return state

    # --- Apply advanced propagation to initial state ---
    w0 = _advanced(w0)
    if w0 is None:
        search = GeneralConstructiveSearch(
            w0={}, succ=lambda n: [], goal=lambda n: False
        )
        search.OPEN = []
        return search, lambda n: n

    # --- MRV variable selection ---
    def _select_var(state):
        best = None
        best_count = 17
        for var, mask in state.items():
            if mask & (mask - 1):
                c = _popcount(mask)
                if c < best_count:
                    best_count = c
                    best = var
                    if c == 2:
                        return best
        return best

    use_phi = phi is not None

    # --- Successor function ---
    def succ(state):
        if use_phi:
            doms = {var: _from_bitmask(mask) for var, mask in state.items()}
            var = phi(doms)
        else:
            var = _select_var(state)
        if var is None:
            return []
        mask = state[var]
        successors = []
        bits = mask
        while bits:
            bit = bits & -bits
            bits &= bits - 1
            ns = state.copy()
            ns[var] = bit
            ns = _propagate(ns, var)
            if ns is None:
                continue
            ns = _advanced(ns)
            if ns is None:
                continue
            successors.append(ns)
        return successors

    # --- Goal test ---
    def goal(state):
        for mask in state.values():
            if mask & (mask - 1):
                return False
        return True

    # --- Decoder ---
    def decoder(state):
        return {var: state[var].bit_length() for var in state}

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
    _neq = lambda a, b: a != b
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            constraints[(variables[i], variables[j])] = _neq
    return constraints
