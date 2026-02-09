from task_encodings import get_general_constructive_search_for_jobshop
import numpy as np

def test_jobshop():
    # 2 machines, jobs that take 10, 20, and 30 minutes
    num_machines = 2
    durations = [10, 20, 30]
    jobshop_data = (num_machines, durations)

    print(f"Testing JobShop: {num_machines} machines, Jobs: {durations}")

    # 1. Get the engine and decoder
    search, decoder = get_general_constructive_search_for_jobshop(jobshop_data)

    # 2. Run the search
    # Note: Since we haven't implemented 'better' yet, it might just find the FIRST valid one
    print("Running search...")
    while search.active:
        search.step()

    # 3. Decode result
    if search.best:
        result = decoder(search.best)
        print("\nFinal Assignment (Job IDs per Machine):")
        for m_id, jobs in result.items():
            print(f"Machine {m_id}: Jobs {jobs}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    test_jobshop()