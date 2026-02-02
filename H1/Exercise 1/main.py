from connect_state import ConnectState
import numpy as np

def test_environment():
    # 0. Optional setup, debug board (0...41)
    position_map = np.arange(42).reshape(6, 7)

    # 1. Initialize the State (s0)
    print("Initializing Connect Four Environment...")
    state = ConnectState(position_map)
    # state = ConnectState()

    # 2. Test initial board setup
    print("Initial Board State:")
    state.show_terminal()

    print(f'HEIGHTS:', state.get_heights())

    for i in range(6):
        print(state.is_col_free(i))


if __name__ == "__main__":
    test_environment()