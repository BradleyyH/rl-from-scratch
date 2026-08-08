"""Run Q-learning over 5 different seeds and report mean and standard deviation."""

import numpy as np

from algorithms.q_learning import evaluate, train

SEEDS = [42, 50, 100, 1000, 1607]

def run_experiments() -> None:
    """Train and evaluate Q-learning across multiple seeds"""
    success_rates=[]

    for seed in SEEDS:
        print(f"\n Seed {seed}")
        q_table = train(seed=seed)
        success_rate = evaluate(q_table, seed=seed)
        success_rates.append(success_rate)

    success_rates = np.array(success_rates)
    print("\n Results")
    print(f"Seeds: {SEEDS}")
    print(f"Success rates: {[f'{r:.1%}' for r in success_rates]}")
    print(f"Mean: {np.mean(success_rates):.1%}")
    print(f"Std:  {np.std(success_rates):.1%}")

if __name__ == "__main__":
    run_experiments()