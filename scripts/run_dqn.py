"""Run DQN over 5 different seeds and report mean and standard deviation."""

import matplotlib.pyplot as plt
import numpy as np

from algorithms.dqn import evaluate, train

SEEDS = [42, 50, 100, 1000, 1607]

def plot_mean_rewards(mean_rewards: np.ndarray) -> None:
    """Plot mean reward across seeds."""
    fig, ax = plt.subplots(figsize=(6, 4))
    seeds = list(range(1, len(mean_rewards) + 1))
    mean = np.mean(mean_rewards)
    std = np.std(mean_rewards)

    ax.bar(seeds, mean_rewards, color="blue", alpha=0.7, label="Mean reward per seed")
    ax.axhline(mean, color="red", linestyle="--", label=f"Mean: {mean:.1f}")
    ax.fill_between([0.5, len(seeds) + 0.5], mean - std, mean + std, alpha=0.2, color="red", label=f"±1 std: {std:.1f}")
    ax.set_xlabel("Seed index")
    ax.set_ylabel("Mean reward")
    ax.set_title("DQN on CartPole-v1 with 5 seeds")
    ax.set_ylim(0, 500)
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/dqn_mean_rewards.png")
    plt.close()


def plot_learning_curves(all_logs: list) -> None:
    """Plot rolling mean reward across all seeds."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for seed, log in zip(SEEDS, all_logs):
        episodes, rewards = zip(*log)
        ax.plot(episodes, rewards, alpha=0.7, label=f"Seed {seed}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rolling mean reward")
    ax.set_title("DQN Learning Curves on CartPole-v1")
    ax.set_ylim(0, 500)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/dqn_learning_curves.png")
    plt.close()


def run_experiments() -> None:
    """Train and evaluate DQN across multiple seeds."""
    mean_rewards = []
    all_logs = []

    for i, seed in enumerate(SEEDS):
        print("\n Seed", seed)
        net, rolling_log = train(seed=seed)
        mean_reward = evaluate(net, seed=seed)
        mean_rewards.append(mean_reward)
        all_logs.append(rolling_log)

    mean_rewards = np.array(mean_rewards)
    print("\n Results")
    print(f"Seeds: {SEEDS}")
    print(f"Mean rewards: {[f'{r:.1f}' for r in mean_rewards]}")
    print(f"Mean: {np.mean(mean_rewards):.1f}")
    print(f"Std:  {np.std(mean_rewards):.1f}")

    plot_mean_rewards(mean_rewards)
    plot_learning_curves(all_logs)


if __name__ == "__main__":
    run_experiments()