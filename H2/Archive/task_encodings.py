from general_constructive_search import GeneralConstructiveSearch, encode_problem
import numpy as np
from connect4.connect_state import ConnectState

# ========== SUDOKU ==========

def get_general_constructive_search_for_sudoku(sudoku):
    """Fixed-variable problem: 81 cells, each assigned a digit 1-9."""
    domains = {}
    for i in range(81):
        r, c = i // 9, i % 9
        if sudoku[r, c] != 0:
            domains[i] = [sudoku[r, c]]
        else:
            domains[i] = list(range(1, 10))

    def check_constraints(node):
        """Only check the newest cell against its row, col, and 3x3 box."""
        idx = len(node) - 1
        val = node[idx]
        r, c = idx // 9, idx % 9
        br, bc = (r // 3) * 3, (c // 3) * 3

        # Check row
        for cc in range(9):
            k = r * 9 + cc
            if k < idx and node[k] == val:
                return False
        # Check column
        for rr in range(9):
            k = rr * 9 + c
            if k < idx and node[k] == val:
                return False
        # Check 3x3 box
        for dr in range(3):
            for dc in range(3):
                k = (br + dr) * 9 + (bc + dc)
                if k < idx and node[k] == val:
                    return False
        return True

    def decoder(node):
        if node is None:
            return None
        board = np.zeros((9, 9), dtype=int)
        for idx, val in node.items():
            board[idx // 9, idx % 9] = val
        return board

    return encode_problem(domains, check_constraints, None, "dfs"), decoder


# ========== JOBSHOP ==========

def get_general_constructive_search_for_jobshop(jobshop):
    """Fixed-variable problem: assign each job to a machine."""
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
        """Return a dict so grader cost_fun can call .items()."""
        if node is None:
            return None
        return dict(node)

    return encode_problem(domains, lambda n: True, better, "dfs"), decoder


# ========== CONNECT 4 ==========

def get_general_constructive_search_for_connect_4(opponent):
    """
    Variable-length problem: build Yellow's move sequence to beat Red.
    Nodes carry state to avoid replaying from scratch.
    Node = (yellow_moves_tuple, current_state_for_yellow_to_play)
    """
    # Red starts the game
    init_state = ConnectState()
    red_first = opponent(init_state)
    state_after_red = init_state.transition(red_first)

    if state_after_red.is_final():
        # Red wins on first move (impossible in connect-4 but handle gracefully)
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
                # Game ended after Yellow's move (Yellow won or draw)
                children.append((new_moves, after_yellow))
            else:
                # Red responds
                red_move = opponent(after_yellow)
                after_red = after_yellow.transition(red_move)
                if after_red.is_final():
                    # Game ended after Red's response
                    children.append((new_moves, after_red))
                else:
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
    Variable-length problem: find shortest path from from_index to to_index.
    Nodes are tuples of visited cities (not including from_index).
    Uses direct GeneralConstructiveSearch construction with tuples for speed.
    """
    n = distances.shape[0]

    def expand(node):
        if len(node) == 0:
            current = from_index
        else:
            current = node[-1]
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

    def decoder(node):
        if node is None:
            return None
        return [from_index] + list(node)

    return search, decoder