# Abstract
from environment_state import EnvironmentState

# Types
from typing import Optional, List, Any

# Libraries
import numpy as np
import matplotlib.pyplot as plt


class ConnectState(EnvironmentState):
    """
    Environment state representation for the Connect Four game.
    """

    def __init__(self, board: Optional[np.ndarray] = None):
        """
        Initializes the Connect Four game state.

        Parameters
        ----------
        board : Optional[np.ndarray]
            A NumPy array representing the board state. If None, an empty board is created.
        """
        raise NotImplementedError("Class constructor must be implemented.")

    def is_final(self) -> bool:
        """See base class."""
        raise NotImplementedError("Method is_final must be implemented.")

    def is_applicable(self, event: Any) -> bool:
        """See base class."""
        raise NotImplementedError("Method is_applicable must be implemented.")

    def transition(self, col: int) -> "EnvironmentState":
        """See base class."""
        raise NotImplementedError("Method put must be implemented.")

    def get_winner(self) -> int:
        """
        Determines the winner in the current state.

        Returns
        -------
        int
            -1 if red has won, 1 if yellow has won, 0 if no winner.
        """
        raise NotImplementedError("Method get_winner must be implemented.")

    def is_col_free(self, col: int) -> bool:
        """
        Checks if a tile can be placed in the specified column.

        Parameters
        ----------
        col : int
            Index of the column.

        Returns
        -------
        bool
            True if the column has space for a tile; False otherwise.
        """
        raise NotImplementedError("Method is_col_free must be implemented.")

    def get_heights(self) -> List[int]:
        """
        Gets the number of tiles placed in each column.

        Returns
        -------
        List[int]
            A list of integers indicating the number of tiles per column.
        """
        raise NotImplementedError("Method get_heights must be implemented.")

    def get_free_cols(self) -> List[int]:
        """
        Gets the list of columns where a tile can still be placed.

        Returns
        -------
        List[int]
            Indices of columns with at least one free cell.
        """
        raise NotImplementedError("Method get_free_cols must be implemented.")

    def show(self, size: int = 1500, ax: Optional[plt.Axes] = None) -> None:
        """
        Visualizes the current board state using matplotlib.

        Parameters
        ----------
        size : int, optional
            Size of the stones, by default 1500.
        ax : Optional[matplotlib.axes._axes.Axes], optional
            Axes to plot on. If None, a new figure is created.
        """
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
