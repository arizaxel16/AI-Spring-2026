from general_constructive_search import GeneralConstructiveSearch, encode_problem
import numpy as np
from connect4.connect_state import ConnectState

# ========== SUDOKU ==========

def get_general_constructive_search_for_sudoku(sudoku):
    """
    Prepares a GeneralConstructiveSearch to solve a Sudoku puzzle.
    """
    # 1. BUILD DOMAINS
    # We map the 2D sudoku input into a 1D 'flat' dictionary for the engine
    domains = {}
    for r in range(9):
        for c in range(9):
            flat_idx = r * 9 + c
            if sudoku[r, c] == 0:
                domains[flat_idx] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            else:
                # If it's a pre-filled number, the search only has one choice
                domains[flat_idx] = [sudoku[r, c]]

    # 2. CHOOSE STRATEGY
    order = "bfs"

    # 3. CREATE SEARCH OBJECT
    # We pass our generic engine the specific Sudoku constraints
    search_obj = encode_problem(domains, check_sudoku_constraints, None, order)

    # 4. RETURN BOTH
    # The grader needs the engine to RUN the search
    # and the decoder to READ the result.
    return search_obj, sudoku_decoder

# ========== SUDOKU AUX ==========

def is_valid_sudoku_row(val, row, prev_val, prev_row):
    """Returns True if there is no row conflict."""
    if (row == prev_row) and (val == prev_val):
        return False
    else:
        return True

def is_valid_sudoku_col(val, col, prev_val, prev_col):
    """Returns True if there is no column conflict."""
    if (col == prev_col) and (val == prev_val):
        return False
    else:
        return True

def is_valid_sudoku_3by3(val, row, col, prev_val, prev_row, prev_col):
    """Returns True if there is no 3x3 box conflict."""
    if (row//3 == prev_row//3) and (col//3 == prev_col//3) and (val == prev_val):
        return False
    else:
        return True

def check_sudoku_constraints(node):
    """The 'Referee' that combines the micro-rules."""
    # 1. Identify the new piece of the puzzle
    val = node[-1]
    idx = len(node) - 1
    row = idx // 9
    col = idx % 9

    # 2. Compare against all previous pieces
    for i in range(len(node) - 1):
        prev_val = node[i]
        prev_row = i // 9
        prev_col = i % 9

        # 3. Check all three rules
        # If ANY rule is violated, return False immediately
        if not is_valid_sudoku_row(val, row, prev_val, prev_row):
            return False

        if not is_valid_sudoku_col(val, col, prev_val, prev_col):
            return False

        if not is_valid_sudoku_3by3(val, row, col, prev_val, prev_row, prev_col):
            return False

    return True

def sudoku_decoder(node):
    # Create a blank 9x9 board
    import numpy as np
    board = np.zeros((9, 9), dtype=int)

    # Fill it up
    for i in range(len(node)):
        r = i // 9
        c = i % 9
        board[r, c] = node[i]
    return board

# ========== JOBSHOP ==========

def get_general_constructive_search_for_jobshop(jobshop):
    """
    Encodes a Job Shop Scheduling problem as a GeneralConstructiveSearch.

    Args:
        jobshop (tuple): A tuple (m, d) where:
            - m (int): Number of machines.
            - d (list): List of job durations.

    Returns:
        tuple: (GeneralConstructiveSearch, decoder)
            - search: Search object for solving the job shop.
            - decoder: Function to decode final node into job-machine assignments.
    """
    # 1. BUILD DOMAINS
    # We map the job machine-duration pairs a 1D 'flat' dictionary for the engine
    num_machines, durations = jobshop

    # We need a list of machine IDs: [0, 1, 2...]
    machine_options = list(range(num_machines))

    # Keys are job indices (0 to len(durations)-1)
    domains = {i: machine_options for i in range(len(durations))}

    # 2. CHOOSE STRATEGY
    order = "dfs"

    def better(node1, node2):
        return calculate_makespan(node1, durations, num_machines) < calculate_makespan(node2, durations, num_machines)

    # 3. CREATE SEARCH OBJECT
    search_obj = encode_problem(domains, check_jobshop_constraints, better, order)

    # 4. RETURN BOTH
    return search_obj, jobshop_decoder

# ========== JOBSHOP AUX ==========

def calculate_makespan(node, durations, num_machines):
    clocks = [0] * num_machines
    # We loop through the decisions made in the node so far
    for job_idx, machine_id in enumerate(node):
        clocks[machine_id] += durations[job_idx]
    return max(clocks)

def check_jobshop_constraints(node):
    return True

def jobshop_decoder(node):
    # It maps Machine ID -> List of Job IDs
    machine_dist = {}
    for job_id, machine_id in enumerate(node):
        machine_dist.setdefault(machine_id, []).append(job_id)
    return machine_dist

# ========== CONNECT 4 ==========

def get_general_constructive_search_for_connect_4(opponent):
    """
    Creates a GeneralConstructiveSearch to find a winning strategy for Connect-4.

    Args:
        opponent (Callable): Function mapping state to opponent's move.

    Returns:
        tuple: (GeneralConstructiveSearch, decoder)
            - search: Search object to solve the game.
            - decoder: Function to extract yellow player’s move sequence.

    Note:
        You may choose DFS or BFS with the order argument.
    """
    raise NotImplementedError(
        "You must implement 'get_general_constructive_search_for_connect_4'"
    )


def get_general_constructive_search_for_tour_planning(distances, from_index, to_index):
    """
    Encodes a tour planning problem as a GeneralConstructiveSearch.

    Args:
        distances (np.ndarray): Adjacency matrix of distances between cities.
        from_index (int): Starting city index.
        to_index (int): Target city index.

    Returns:
        tuple: (GeneralConstructiveSearch, decoder)
            - search: Search object to solve the tour planning problem.
            - decoder: Function that returns the full path of the tour.

    Note:
        You may choose DFS or BFS with the order argument.
    """
    raise NotImplementedError(
        "You must implement 'get_general_constructive_search_for_tour_planning'"
    )
