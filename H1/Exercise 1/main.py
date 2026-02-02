import random
from connect_state import ConnectState

def play_random_game():
    # 1. Start at the initial state s0
    state = ConnectState()
    print("--- Game Started ---")
    state.show_terminal()

    # 2. Loop until the referee says it's over
    while not state.is_final():
        current_player = "Red (-1)" if state.get_player() == -1 else "Yellow (1)"

        legal_moves = state.get_free_cols()
        chosen_col = random.choice(legal_moves)

        print(f"\n{current_player} drops a tile in column {chosen_col}")

        # 3. Transition to the next state
        state = state.transition(chosen_col)
        state.show_terminal()

    # 4. Results
    winner = state.get_winner()
    if winner == -1:
        print("\n*** RED WINS! ***")
    elif winner == 1:
        print("\n*** YELLOW WINS! ***")
    else:
        print("\n*** IT'S A DRAW! ***")

    state.show()

if __name__ == "__main__":
    play_random_game()