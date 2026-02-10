from general_constructive_search import GeneralConstructiveSearch, encode_problem
import numpy as np
from connect4.connect_state import ConnectState

# ========== SUDOKU ==========

def get_general_constructive_search_for_sudoku(sudoku):
    """
    Only search empty cells, pre-reduce domains, MRV variable ordering.
    """
    # Precompute all peers for each cell
    all_peers = [set() for _ in range(81)]
    for i in range(81):
        r, c = i // 9, i % 9
        br, bc = (r // 3) * 3, (c // 3) * 3
        for cc in range(9):
            all_peers[i].add(r * 9 + cc)
        for rr in range(9):
            all_peers[i].add(rr * 9 + c)
        for dr in range(3):
            for dc in range(3):
                all_peers[i].add((br + dr) * 9 + (bc + dc))
        all_peers[i].discard(i)

    # Collect given values
    given = {}
    for i in range(81):
        r, c = i // 9, i % 9
        if sudoku[r, c] != 0:
            given[i] = sudoku[r, c]

    # Pre-reduce domains for empty cells by eliminating values from given peers
    empty_cells = []
    cell_domain = {}
    for i in range(81):
        if i in given:
            continue
        used = set()
        for p in all_peers[i]:
            if p in given:
                used.add(given[p])
        cell_domain[i] = [v for v in range(1, 10) if v not in used]
        empty_cells.append(i)

    # MRV: sort empty cells by domain size (most constrained first)
    empty_cells.sort(key=lambda i: len(cell_domain[i]))

    # Precompute: for each empty cell in our order, which prior empty cells are peers
    cell_to_idx = {cell: idx for idx, cell in enumerate(empty_cells)}
    prior_empty_peers = [[] for _ in range(len(empty_cells))]
    for idx, cell in enumerate(empty_cells):
        for p in all_peers[cell]:
            if p in cell_to_idx and cell_to_idx[p] < idx:
                prior_empty_peers[idx].append(cell_to_idx[p])

    n_empty = len(empty_cells)

    def expand(node):
        idx = len(node)
        if idx >= n_empty:
            return []
        children = []
        for val in cell_domain[empty_cells[idx]]:
            valid = True
            for pidx in prior_empty_peers[idx]:
                if node[pidx] == val:
                    valid = False
                    break
            if valid:
                children.append(node + (val,))
        return children

    def goal(node):
        return len(node) == n_empty

    search = GeneralConstructiveSearch(expand, goal, None, "dfs")
    search.initial = ()
    search.reset()

    def decoder(node):
        if node is None:
            return None
        board = sudoku.copy()
        for idx, val in enumerate(node):
            cell = empty_cells[idx]
            board[cell // 9, cell % 9] = val
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