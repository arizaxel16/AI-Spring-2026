from __future__ import annotations
import json

from lake_mdp import LakeMDP
from policies import RandomPolicy, CustomPolicy
from utility_analyzer import UtilityAnalyzer

DEFAULT_MAP = [
    ['S', 'F', 'F', 'F'],
    ['F', 'H', 'F', 'F'],
    ['F', 'F', 'F', 'F'],
    ['H', 'F', 'F', 'G'],
]

def evaluate_all(trials: int = 10, base_seed: int = 42):
    """
    Evaluate RandomPolicy and CustomPolicy for γ ∈ {0.5, 0.9, 1.0}.
    Returns a JSON-serializable dict with summaries and the winner per γ.
    """
    mdp = LakeMDP(DEFAULT_MAP)
    report = {"n_trials": int(trials), "base_seed": int(base_seed), "gammas": {}}

    for gamma in [0.5, 0.9, 1.0]:
        analyzer = UtilityAnalyzer(mdp, gamma=gamma)

        sum_rand = analyzer.evaluate(RandomPolicy, trials, base_seed)
        sum_cust = analyzer.evaluate(CustomPolicy, trials, base_seed)

        # Tie-breaker logic
        if sum_rand["mean_utility"] > sum_cust["mean_utility"]:
            best = "random"
        elif sum_cust["mean_utility"] > sum_rand["mean_utility"]:
            best = "custom"
        else:
            # Means are equal, check variance
            if sum_rand["utility_variance"] < sum_cust["utility_variance"]:
                best = "random"
            else:
                best = "custom"

        report["gammas"][str(gamma)] = {
            "random": sum_rand,
            "custom": sum_cust,
            "winner": best,
        }

    return report

if __name__ == "__main__":
    # Test the execution
    results = evaluate_all()
    print(json.dumps(results, indent=2))