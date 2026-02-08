import numpy as np
from task_encodings import get_general_constructive_search_for_sudoku

def test_solver():
    # 1. Create a sample Sudoku (0 = empty)
    # This is a standard "Easy" puzzle
    board = np.array([
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

    print("--- Original Board ---")
    print(board)

    # 2. Get the Search Engine and Decoder from your factory
    search, decoder = get_general_constructive_search_for_sudoku(board)

    # 3. Run the search
    print("\nSolving...")
    steps = 0
    while search.active:
        found_new_best = search.step()
        steps += 1
        if steps % 500 == 0:
            print(f"Still searching... {steps} steps taken.")

    # 4. Check results
    if search.best is not None:
        print(f"\nSuccess! Solved in {steps} steps.")
        final_solution = decoder(search.best)
        print("--- Solved Board ---")
        print(final_solution)
    else:
        print("\nNo solution found. Check your constraints logic!")

if __name__ == "__main__":
    test_solver()