import unittest
import numpy as np
import sys
import os

# Manual path fix to ensure it can see task_encodings.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from task_encodings import get_general_constructive_search_for_sudoku

class TestSudoku(unittest.TestCase):
    def setUp(self):
        # A simple valid Sudoku board (0 = empty)
        self.board = np.array([
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ])

    def test_solver_completes(self):
        """Check if the solver finds a valid 9x9 board."""
        search, decoder = get_general_constructive_search_for_sudoku(self.board)

        while search.active:
            search.step()

        self.assertIsNotNone(search.best, "Solver failed to find a solution.")
        solution = decoder(search.best)

        # Assert shape and that no zeros remain
        self.assertEqual(solution.shape, (9, 9))
        self.assertFalse(0 in solution, "Solution contains empty cells.")

    def test_sudoku_validity(self):
        """Check if rows in the solution contain unique digits 1-9."""
        search, decoder = get_general_constructive_search_for_sudoku(self.board)
        while search.active:
            search.step()

        solution = decoder(search.best)
        for row in range(9):
            self.assertEqual(len(set(solution[row, :])), 9, f"Row {row} has duplicate digits.")

if __name__ == "__main__":
    unittest.main()