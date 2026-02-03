# Abstract
from environment_state import EnvironmentState

# Types
from typing import Optional, List

# Libraries
import numpy as np
import matplotlib.pyplot as plt


class ConnectState(EnvironmentState):
    """
    Environment state representation for the Connect Four game.
    """

    ROWS = 6
    COLS = 7

    def __init__(self, board: Optional[np.ndarray] = None):
        """
        Initializes the Connect Four game state.

        Parameters
        ----------
        board : Optional[np.ndarray]
            A NumPy array representing the board state. If None, an empty board is created.
        """
        if board is not None:
            self.board = board
        else:
            self.board = np.zeros((self.ROWS, self.COLS), dtype=int)

    def is_final(self) -> bool:
        if self.get_winner() != 0:
            return True

        if len(self.get_free_cols()) == 0:
            return True

        return False

    def is_applicable(self, event: int) -> bool:
        if not (0 <= event < self.COLS):
            return False
        return self.is_col_free(event)

    def transition(self, col: int) -> "ConnectState":
        if not self.is_applicable(col):
            raise ValueError(f"Action {col} is not applicable in the current state.")

        player = self.get_player()
        row = (self.ROWS - 1) - self.get_heights()[col]

        new_board = self.board.copy()
        new_board[row, col] = player
        return ConnectState(new_board)

    def get_winner(self) -> int:
        board = self.board

        # 1. Horizontal Check
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                window = board[r, c:c+4]
                if np.all(window == 1): return 1
                if np.all(window == -1): return -1

        # 2. Vertical Check
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                window = board[r:r+4, c]
                if np.all(window == 1): return 1
                if np.all(window == -1): return -1

        # 3. Diagonal Check (Negative Slope: \)
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                window = np.array([board[r+i, c+i] for i in range(4)])
                if np.all(window == 1): return 1
                if np.all(window == -1): return -1

        # 4. Diagonal Check (Positive Slope: /)
        for r in range(3, self.ROWS):
            for c in range(self.COLS - 3):
                window = np.array([board[r-i, c+i] for i in range(4)])
                if np.all(window == 1): return 1
                if np.all(window == -1): return -1

        return 0

    def get_player(self) -> int:
        return -1 if np.count_nonzero(self.board) % 2 == 0 else 1

    def is_col_free(self, col: int) -> bool:
        rows, _ = self.board.shape
        return self.get_heights()[col] < rows

    def get_heights(self) -> List[int]:
        return np.count_nonzero(self.board, axis=0).tolist()

    def get_free_cols(self) -> List[int]:
        free_cols = []
        _, num_cols = self.board.shape

        for i in range(num_cols):
            if self.is_col_free(i):
                free_cols.append(i)
        return free_cols

    def show(self, size: int = 1500, ax: Optional[plt.Axes] = None) -> None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        pos_red = np.where(self.board == -1)
        pos_yellow = np.where(self.board == 1)

        ax.scatter(pos_yellow[1] + 0.5, 5.5 - pos_yellow[0], color="yellow", s=size)
        ax.scatter(pos_red[1] + 0.5, 5.5 - pos_red[0], color="red", s=size)

        ax.set_ylim([0, self.board.shape[0]])
        ax.set_xlim([0, self.board.shape[1]])
        ax.grid()

        if fig is not None:
            plt.show()

    def show_terminal(self) -> None:
        print(self.board)
