from general_constructive_search import GeneralConstructiveSearch, encode_problem
import numpy as np
from connect4.connect_state import ConnectState

# ========== SUDOKU ==========

def get_general_constructive_search_for_sudoku(sudoku):
    """
    Bypass encode_problem for performance: tuple nodes, precomputed peers.
    """
    # Precompute peers (cells sharing row, col, or box) with index < i
    prior_peers = [[] for _ in range(81)]
    for i in range(81):
        r, c = i // 9, i % 9
        br, bc = (r // 3) * 3, (c // 3) * 3
        s = set()
        for cc in range(9):
            s.add(r * 9 + cc)
        for rr in range(9):
            s.add(rr * 9 + c)
        for dr in range(3):
            for dc in range(3):
                s.add((br + dr) * 9 + (bc + dc))
        s.discard(i)
        prior_peers[i] = [k for k in s if k < i]

    # Precompute domain for each cell
    cell_domain = []
    for i in range(81):
        r, c = i // 9, i % 9
        if sudoku[r, c] != 0:
            cell_domain.append([sudoku[r, c]])
        else:
            cell_domain.append(list(range(1, 10)))

    def expand(node):
        idx = len(node)
        if idx >= 81:
            return []
        children = []
        peers = prior_peers[idx]
        for val in cell_domain[idx]:
            valid = True
            for k in peers:
                if node[k] == val:
                    valid = False
                    break
            if valid:
                children.append(node + (val,))
        return children

    def goal(node):
        return len(node) == 81

    search = GeneralConstructiveSearch(expand, goal, None, "dfs")
    search.initial = ()
    search.reset()

    def decoder(node):
        if node is None:
            return None
        board = np.zeros((9, 9), dtype=int)
        for i, val in enumerate(node):
            board[i // 9, i % 9] = val
        return board

    return search, decoder


# ========== JOBSHOP ==========

def get_general_constructive_search_for_jobshop(jobshop):
    m, d = jobshop
    n_jobs = len(d)
    domains = {i: list(range(m)) for i in range(n_jobs)}

    def makespan(node):
        clocks = [0] * m
        for job_idx, machine_id in node.items():
            clocks[machine_id] += d[job_idx]
        return max(clocks)

    def better(n1, n2):
        return makespan(n1) < makespan(n2)

    def decoder(node):
        if node is None:
            return None
        return dict(node)

    return encode_problem(domains, lambda n: True, better, "dfs"), decoder


# ========== CONNECT 4 ==========

def get_general_constructive_search_for_connect_4(opponent):
    """
    Nodes carry state to avoid replaying: (yellow_moves_tuple, state).
    """
    init_state = ConnectState()
    red_first = opponent(init_state)
    state_after_red = init_state.transition(red_first)

    if state_after_red.is_final():
        search = GeneralConstructiveSearch(lambda n: [], lambda n: False, None, "dfs")
        return search, lambda n: None

    def expand(node):
        moves, state = node
        if state.is_final():
            return []
        children = []
        for col in state.get_free_cols():
            after_yellow = state.transition(col)
            new_moves = moves + (col,)
            if after_yellow.is_final():
                children.append((new_moves, after_yellow))
            else:
                red_move = opponent(after_yellow)
                after_red = after_yellow.transition(red_move)
                children.append((new_moves, after_red))
        return children

    def goal(node):
        moves, state = node
        return state.get_winner() == 1

    search = GeneralConstructiveSearch(expand, goal, None, "dfs")
    search.initial = ((), state_after_red)
    search.reset()

    def decoder(node):
        if node is None:
            return None
        moves, state = node
        return list(moves)

    return search, decoder


# ========== TOUR PLANNING ==========

def get_general_constructive_search_for_tour_planning(distances, from_index, to_index):
    """
    Tuple nodes for speed. Bypass encode_problem.
    """
    n = distances.shape[0]

    def expand(node):
        current = node[-1] if len(node) > 0 else from_index
        visited = set(node)
        visited.add(from_index)
        children = []
        for city in range(n):
            if city in visited:
                continue
            d = distances[current, city]
            if d == 0 or np.isinf(d):
                continue
            children.append(node + (city,))
        return children

    def goal(node):
        return len(node) > 0 and node[-1] == to_index

    def calc_dist(node):
        total, curr = 0, from_index
        for city in node:
            total += distances[curr, city]
            curr = city
        return total

    def better(n1, n2):
        return calc_dist(n1) < calc_dist(n2)

    search = GeneralConstructiveSearch(expand, goal, better, "dfs")
    search.initial = ()
    search.reset()

    def decoder(node):
        if node is None:
            return None
        return [from_index] + list(node)

    return search, decoder