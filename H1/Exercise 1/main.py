from connect_state import ConnectState
import numpy as np

def test_environment():
    # 0. Optional setup, debug board (0...41)
    position_map = np.arange(42).reshape(6, 7)

    # 1. Initialize the State (s0)
    print("Initializing Connect Four Environment...")
    # state = ConnectState(position_map)
    state = ConnectState()
    rows, cols = state.board.shape

    # 2. Test initial board setup
    print("Initial Board State:")
    state.show_terminal()

    print(f'HEIGHTS:', state.get_heights())

    for i in range(cols):
        print(f'COL {i} FREE: {state.is_col_free(i)}')

    print(f'FREE COLS:', state.get_free_cols())



if __name__ == "__main__":
    test_environment()