from general_constructive_search import GeneralConstructiveSearch, encode_problem
import numpy as np
from connect4.connect_state import ConnectState

# ========== SUDOKU ==========

def get_general_constructive_search_for_sudoku(sudoku):
    """
    Prepares a GeneralConstructiveSearch to solve a Sudoku puzzle.

    Args:
        sudoku (np.ndarray): 9x9 numpy array representing the Sudoku board.

    Returns:
        tuple: (GeneralConstructiveSearch, decoder)
            - search: Search object for solving the Sudoku.
            - decoder: Function to decode final node to 9x9 board.

    Note:
        You may choose DFS or BFS with the order argument.
    """
    raise NotImplementedError(
        "You must implement 'get_general_constructive_search_for_sudoku'"
    )

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
    if (row/3 == prev_row/3) and (col/3 == prev_col/3) and (val == prev_val):
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

def get_general_constructive_search_for_sudoku(sudoku):
    """
    This is the "Factory". It builds the rules and hands them to your Engine.

    Args:
        sudoku: A 9x9 numpy array (0 for empty, 1-9 for fixed)
    """

    # 1. BUILD DOMAINS
    # Create a dictionary 'domains' with keys 0 to 80.
    # Logic: Loop through rows (0-8) and cols (0-8).
    # Flat Index = r * 9 + c
    # If sudoku[r,c] == 0: domain is [1,2,3,4,5,6,7,8,9]
    # If sudoku[r,c] != 0: domain is [sudoku[r,c]]
    domains = {}

    # 2. CHOOSE STRATEGY
    # Usually "dfs" is better for Sudoku to find the first solution quickly.
    order = "dfs"

    # 3. CALL YOUR BRIDGE (Exercise 2)
    # Use the 'encode_problem' function you already wrote.
    # Pass: domains, check_sudoku_constraints, None (no 'better' func needed), and order.

    # return encode_problem(...)
    pass

def print_sudoku_solution(node):
    """
    Helpful for you to see if it worked!
    1. Create a blank 9x9 matrix.
    2. Loop through the tuple 'node' and fill the matrix using //9 and %9.
    3. Print the matrix.
    """
    pass

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
    raise NotImplementedError(
        "You must implement 'get_general_constructive_search_for_jobshop'"
    )


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
