import numpy as np, time
from sudoku import get_bcn_for_sudoku
from ac import ac3, _detect_sudoku_groups
from collections import Counter

board_16 = np.array([
    [0,15,0,0,0,0,0,8,6,0,0,0,0,3,0,12],
    [0,0,0,13,0,10,0,0,0,0,15,0,14,0,0,0],
    [9,0,0,0,0,0,0,12,16,0,0,0,0,0,0,5],
    [0,0,0,6,0,0,3,0,0,14,0,0,8,0,0,0],
    [0,0,0,0,2,0,0,0,0,0,0,10,0,0,0,0],
    [3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13],
    [0,0,5,0,0,14,0,0,0,0,3,0,0,11,0,0],
    [0,16,0,0,0,0,0,11,2,0,0,0,0,0,5,0],
    [0,5,0,0,0,0,0,3,11,0,0,0,0,0,2,0],
    [0,0,3,0,0,8,0,0,0,0,14,0,0,5,0,0],
    [13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3],
    [0,0,0,0,10,0,0,0,0,0,0,2,0,0,0,0],
    [0,0,0,8,0,0,6,0,0,10,0,0,11,0,0,0],
    [5,0,0,0,0,0,0,16,12,0,0,0,0,0,0,9],
    [0,0,0,14,0,13,0,0,0,0,8,0,6,0,0,0],
    [12,0,6,0,0,0,0,2,5,0,0,0,0,0,3,0],
])

bcn = get_bcn_for_sudoku(board_16)
reduced, feasible = ac3(bcn)
domains = reduced[0]

def to_bitmask(vals):
    m = 0
    for v in vals:
        m |= 1 << (v - 1)
    return m

state = {var: to_bitmask(vals) for var, vals in domains.items()}
singles = sum(1 for m in state.values() if not (m & (m - 1)))
print(f"After AC-3: {singles} singletons")
sizes = Counter(bin(m).count('1') for m in state.values())
print(f"Domain sizes: {dict(sorted(sizes.items()))}")

neighbors = {var: set() for var in domains}
for (A,B) in bcn[1]:
    neighbors[A].add(B)
    neighbors[B].add(A)
neighbors = {v: list(n) for v,n in neighbors.items()}

def propagate(state, fixed_var):
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

# Propagate all initial singletons
for var, m in list(state.items()):
    if m and not (m & (m-1)):
        result = propagate(state, var)
        if result is None:
            print("INFEASIBLE during initial propagation!")
            import sys; sys.exit(1)

singles = sum(1 for m in state.values() if m and not (m & (m - 1)))
print(f"After singleton propagation: {singles} singletons")
sizes = Counter(bin(m).count('1') for m in state.values())
print(f"Domain sizes: {dict(sorted(sizes.items()))}")

groups = _detect_sudoku_groups(domains)
print(f"Groups detected: {len(groups)}")

changed = True
rounds = 0
while changed:
    changed = False
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
                print(f"ERROR: value {bit.bit_length()} has 0 candidates in group")
                break
            if cnt == 1 and state[candidate] != bit:
                state[candidate] = bit
                state = propagate(state, candidate)
                if state is None:
                    print("INFEASIBLE during hidden singles!")
                    import sys; sys.exit(1)
                changed = True
                rounds += 1
                break
        if changed:
            break

singles = sum(1 for m in state.values() if m and not (m & (m - 1)))
print(f"After hidden singles ({rounds} rounds): {singles} singletons")
sizes = Counter(bin(m).count('1') for m in state.values())
print(f"Domain sizes: {dict(sorted(sizes.items()))}")
