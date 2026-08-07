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

    # Epsilon decay schedule
    decay_episodes = int(N_EPISODES * EPSILON_DECAY)
    epsilon = EPSILON_START

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample() # Exploration
            else:
                action = int(np.argmax(q_table[state])) # Exploitation

            # Take action, observe the reward, and next state
            next_state, reward, terminated, truncated, _= env.step(action)
            done = terminated or truncated # End if episode ends naturally (goal or fell in hole) or time limit is reached

            # Q-learning update (Bellman Equation)
            # If fall in hole, episode terminates, so there is no future value, so zero it out: (1 - terminated)
            td_error = reward + GAMMA * np.max(q_table[next_state]) * (1 - terminated) - q_table[state, action]
            q_table[state, action] += ALPHA * td_error

            state = next_state
            total_reward += reward

        # Decay epsilon linearly
        if episode < decay_episodes:
            epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * (episode / decay_episodes)
        else:
            epsilon = EPSILON_END

        # Log every 500 episodes to W&B to show progression
        if episode % 500 == 0:
            print(f"Episode {episode}, Epsilon:{epsilon:.3f}, Reward: {total_reward}")
            wandb.log({"episode": episode, "reward": total_reward, "epsilon": epsilon})

    env.close()
    wandb.finish()

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