from connect_state import ConnectState
import numpy as np

def test_environment():
    # 1. Initialize the State (s0)
    print("Initializing Connect Four Environment...")
    state = ConnectState()

    # 2. Test initial board setup
    print("Initial Board State:")
    state.show_terminal()


if __name__ == "__main__":
    test_environment()