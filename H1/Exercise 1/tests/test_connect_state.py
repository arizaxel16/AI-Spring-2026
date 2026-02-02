import unittest
import numpy as np
from ..connect_state import ConnectState

class TestConnectFour(unittest.TestCase):

    def test_initialization(self):
        """Ensure the board starts empty."""
        state = ConnectState()
        self.assertEqual(np.sum(state.board), 0)
        self.assertEqual(state.board.shape, (6, 7))

    def test_gravity(self):
        """Ensure tiles stack in a column."""
        state = ConnectState()
        state = state.transition(3) # Red at bottom (row 5)
        state = state.transition(3) # Yellow on top (row 4)
        self.assertEqual(state.board[5, 3], -1)
        self.assertEqual(state.board[4, 3], 1)

    def test_horizontal_win(self):
        """Force a horizontal win and check get_winner."""
        board = np.zeros((6, 7), dtype=int)
        board[5, 0:4] = 1 # Four Yellows in the bottom row
        state = ConnectState(board)
        self.assertEqual(state.get_winner(), 1)
        self.assertTrue(state.is_final())

    def test_invalid_move(self):
        """Test if we can detect a full column."""
        # Create a full column
        board = np.zeros((6, 7), dtype=int)
        board[:, 0] = 1
        state = ConnectState(board)
        self.assertFalse(state.is_applicable(0))