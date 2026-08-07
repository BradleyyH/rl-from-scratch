"""Tabular Q-Learning on FrozenLake-v1."""

import numpy as np
import gymnasium as gym
import wandb
from common.seed import set_seed

# Hyperparameters
ENV_ID = "FrozenLake-v1"
MAP_SIZE = 8
N_EPISODES = 30_000
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.8
SEED = 1607

def train(seed: int = SEED) -> None:
    """Train Q-learning agent on FrozenLake."""
    set_seed(seed)

    # Initialise environment
    env = gym.make(ENV_ID, is_slippery=True, map_name=f"{MAP_SIZE}x{MAP_SIZE}") # 64 rows (8x8)
    set_seed(seed, env)

    # Initialise Q-table with zeros
    # (.n is an attribute of Gymnasium's Discrete space class to give number of possible values)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))

    # Initialise W&B - Trace every result back to its exact configuration
    wandb.init(
        project="rl-from-scratch",
        name=f"q-learning-seed-{seed}",
        config={
            "algorithm": "Q-learning",
            "env": ENV_ID,
            "seed": seed,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "epsilon_decay": EPSILON_DECAY,
            "n_episodes": N_EPISODES
        }
    )