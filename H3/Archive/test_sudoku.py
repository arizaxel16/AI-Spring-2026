import numpy as np
import time
from sudoku import get_bcn_for_sudoku
from ac import get_general_constructive_search_for_bcn, ac3

# --- 9x9 test ---
board_9 = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
])

print("=== 9x9 Sudoku ===")
t0 = time.time()
bcn = get_bcn_for_sudoku(board_9)
search, decoder = get_general_constructive_search_for_bcn(bcn)
while search.active:
    search.step()
elapsed = time.time() - t0
sol = decoder(search.best)
print(f"Solved in {elapsed:.4f}s")

grid = np.zeros((9, 9), dtype=int)
for (r, c), v in sol.items():
    grid[r, c] = v
print(grid)
for i in range(9):
    assert len(set(grid[i, :])) == 9, f"Row {i} invalid"
    assert len(set(grid[:, i])) == 9, f"Col {i} invalid"
print("9x9 PASSED\n")

# --- 16x16 test (known valid puzzle from web) ---
board_16 = np.array([
    [ 0, 15, 0,  0,  0,  0, 0,  8,  6,  0,  0,  0,  0,  3, 0, 12],
    [ 0,  0, 0, 13,  0, 10, 0,  0,  0,  0, 15,  0, 14,  0, 0,  0],
    [ 9,  0, 0,  0,  0,  0, 0, 12, 16,  0,  0,  0,  0,  0, 0,  5],
    [ 0,  0, 0,  6,  0,  0, 3,  0,  0, 14,  0,  0,  8,  0, 0,  0],
    [ 0,  0, 0,  0,  2,  0, 0,  0,  0,  0,  0, 10,  0,  0, 0,  0],
    [ 3,  0, 0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0, 0, 13],
    [ 0,  0, 5,  0,  0, 14, 0,  0,  0,  0,  3,  0,  0, 11, 0,  0],
    [ 0, 16, 0,  0,  0,  0, 0, 11,  2,  0,  0,  0,  0,  0, 5,  0],
    [ 0,  5, 0,  0,  0,  0, 0,  3, 11,  0,  0,  0,  0,  0, 2,  0],
    [ 0,  0, 3,  0,  0,  8, 0,  0,  0,  0, 14,  0,  0,  5, 0,  0],
    [13,  0, 0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0, 0,  3],
    [ 0,  0, 0,  0, 10,  0, 0,  0,  0,  0,  0,  2,  0,  0, 0,  0],
    [ 0,  0, 0,  8,  0,  0, 6,  0,  0, 10,  0,  0, 11,  0, 0,  0],
    [ 5,  0, 0,  0,  0,  0, 0, 16, 12,  0,  0,  0,  0,  0, 0,  9],
    [ 0,  0, 0, 14,  0, 13, 0,  0,  0,  0,  8,  0,  6,  0, 0,  0],
    [12,  0, 6,  0,  0,  0, 0,  2,  5,  0,  0,  0,  0,  0, 3,  0],
])

print("=== 16x16 Sudoku ===")
t0 = time.time()
bcn = get_bcn_for_sudoku(board_16)

# First check AC-3 alone
reduced, feasible = ac3(bcn)
print(f"AC-3 feasible: {feasible}")
if feasible:
    singles = sum(1 for v in reduced[0].values() if len(v) == 1)
    print(f"After AC-3: {singles}/256 cells fixed")

search, decoder = get_general_constructive_search_for_bcn(bcn)
steps = 0
while search.active:
    search.step()
    steps += 1
elapsed = time.time() - t0

if search.best is None:
    print(f"No solution found in {elapsed:.4f}s ({steps} steps)")
else:
    sol = decoder(search.best)
    print(f"Solved in {elapsed:.4f}s  ({steps} steps)")
    grid = np.zeros((16, 16), dtype=int)
    for (r, c), v in sol.items():
        grid[r, c] = v
    print(grid)
    for i in range(16):
        assert len(set(grid[i, :])) == 16, f"Row {i} invalid"
        assert len(set(grid[:, i])) == 16, f"Col {i} invalid"
    print("16x16 PASSED")
