import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from task_encodings import get_general_constructive_search_for_jobshop

class TestJobShop(unittest.TestCase):
    def setUp(self):
        # 2 machines, 3 jobs with durations [10, 20, 30]
        # Perfect balance should be 30 (one machine takes 10+20, other takes 30)
        self.num_machines = 2
        self.durations = [10, 20, 30]
        self.data = (self.num_machines, self.durations)

    def test_optimization(self):
        """Check if JobShop finds the optimal makespan of 30."""
        search, decoder = get_general_constructive_search_for_jobshop(self.data)

        while search.active:
            search.step()

        self.assertIsNotNone(search.best)

        # Calculate makespan from the result
        node = search.best
        clocks = [0] * self.num_machines
        for i, m_id in enumerate(node):
            clocks[m_id] += self.durations[i]

        makespan = max(clocks)
        self.assertEqual(makespan, 30, f"Optimal makespan should be 30, got {makespan}")

    def test_decoder_format(self):
        """Ensure decoder returns a dictionary of machine assignments."""
        search, decoder = get_general_constructive_search_for_jobshop(self.data)
        while search.active:
            search.step()

        result = decoder(search.best)
        self.assertIsInstance(result, dict)
        self.assertTrue(all(isinstance(v, list) for v in result.values()))

if __name__ == "__main__":
    unittest.main()