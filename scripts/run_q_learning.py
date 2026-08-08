"""Run Q-learning over 5 different seeds and report mean and standard deviation."""

import matplotlib.pyplot as plt
import numpy as np

from algorithms.q_learning import evaluate, train, MAP_SIZE

SEEDS = [42, 50, 100, 1000, 1607]

def plot_success_rates(success_rates: np.ndarray) -> None:
    """Plot mean and std bands across seeds."""
    fig, ax = plt.subplots(figsize=(6, 4))
    seeds = list(range(1, len(success_rates) + 1))
    mean = np.mean(success_rates)
    std = np.std(success_rates)

    ax.bar(seeds, success_rates, color="blue", alpha=0.7, label="Success rate across seeds")
    ax.axhline(mean, color="red", linestyle="--", label=f"Mean: {mean:.1%}")
    ax.fill_between([0.5, len(seeds) + 0.5], mean - std, mean + std, alpha=0.2, color="red", label=f"±1 std: {std:.1%}")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Success rate")
    ax.set_title("Q-learning on FrozenLake-v1 with 5 seeds")
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/q_learning_success_rates.png")
    plt.close()

def plot_heatmap(q_table: np.ndarray) -> None:
    """Plot max Q-value per state as a heatmap over the 8x8 grid."""
    fig, ax = plt.subplots(figsize=(8, 8))
    values = np.max(q_table, axis=1).reshape(MAP_SIZE, MAP_SIZE)

    im = ax.imshow(values, cmap="RdYlGn") # Red yellow green
    plt.colorbar(im, ax=ax, label="Max Q-value")
    ax.set_title("Q-table Heatmap on FrozenLake-v1")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.tight_layout()
    plt.savefig("results/q_learning_heatmap.png")
    plt.close()

def run_experiments() -> None:
    """Train and evaluate Q-learning across multiple seeds"""
    success_rates=[]

    last_q_table = None
    for seed in SEEDS:
        print(f"\n Seed {seed}")
        q_table = train(seed=seed)
        success_rate = evaluate(q_table, seed=seed)
        success_rates.append(success_rate)
        last_q_table = q_table

    success_rates = np.array(success_rates)
    print("\n Results")
    print(f"Seeds: {SEEDS}")
    print(f"Success rates: {[f'{r:.1%}' for r in success_rates]}")
    print(f"Mean: {np.mean(success_rates):.1%}")
    print(f"Std:  {np.std(success_rates):.1%}")

    plot_success_rates(success_rates)
    plot_heatmap(last_q_table)

if __name__ == "__main__":
    run_experiments()