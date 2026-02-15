import numpy as np
from ac import get_binarized_constraints_for_all_diff


def get_bcn_for_sudoku(sudoku):
    """
        Receives a Sudoku and creates a BCN definition from it, using the binarized All-Diff Constraint

    Args:
        sudoku (np.ndarray): numpy array describing the sudoku

    Returns:
        (domains, constraints): BCN describing the conditions for the given Sudoku
    """
    n = sudoku.shape[0]
    block_size = int(np.sqrt(n))
    all_values = set(range(1, n + 1))

    # Build domains: fixed cells get singleton, empty cells get full range
    domains = {}
    for i in range(n):
        for j in range(n):
            val = int(sudoku[i, j])
            if val != 0:
                domains[(i, j)] = {val}
            else:
                domains[(i, j)] = set(all_values)

    # Build all-diff constraints for rows, columns, and blocks
    constraints = {}
    for r in range(n):
        row_doms = {(r, c): domains[(r, c)] for c in range(n)}
        constraints.update(get_binarized_constraints_for_all_diff(row_doms))
    for c in range(n):
        col_doms = {(r, c): domains[(r, c)] for r in range(n)}
        constraints.update(get_binarized_constraints_for_all_diff(col_doms))
    for bi in range(block_size):
        for bj in range(block_size):
            blk_doms = {}
            for r in range(bi * block_size, (bi + 1) * block_size):
                for c in range(bj * block_size, (bj + 1) * block_size):
                    blk_doms[(r, c)] = domains[(r, c)]
            constraints.update(get_binarized_constraints_for_all_diff(blk_doms))

    return domains, constraints
