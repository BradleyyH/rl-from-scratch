"""Tabular Q-Learning on FrozenLake-v1."""

import gymnasium as gym
import imageio
import numpy as np

import wandb
from common.seed import set_seed

# Hyperparameters
ENV_ID = "FrozenLake-v1"
MAP_SIZE = 8
N_EPISODES = 100_000
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.8
SEED = 1607

def save_gif(q_table: np.ndarray, seed: int, episode: int, epsilon: float) -> None:
    """Save a GIF of an episode using the current Q-table."""

    env = gym.make(ENV_ID, is_slippery=True, map_name=f"{MAP_SIZE}x{MAP_SIZE}", render_mode="rgb_array")
    state, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)  # local RNG as not to take form the global np.random stream used by the train function.
    frames = []
    done = False

    while not done:
        frames.append(env.render())
        if rng.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q_table[state]))
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    frames.append(env.render())
    env.close()

    path = f"results/q_learning_episode_{episode}_seed_{seed}.gif"
    imageio.mimsave(path, frames, fps=4)
    print(f"Saved GIF: {path}")

def train(seed: int = SEED) -> tuple[np.ndarray, list[float]]:
    """Train Q-learning agent on FrozenLake."""
    set_seed(seed)

    # Initialise environment
    env = gym.make(ENV_ID, is_slippery=True, map_name=f"{MAP_SIZE}x{MAP_SIZE}") # 64 rows (8x8)
    set_seed(seed, env)
    env.action_space.seed(seed)

    # Initialise Q-table with zeros
    # (.n is an attribute of Gymnasium's Discrete space class to give number of possible values)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))

    # Initialise W&B - Trace every result back to its exact configuration
    wandb.init(
        project="rl-from-scratch",
        name=f"q-learning-seed-{seed}",
        reinit=True,
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

    # Epsilon decay schedule
    decay_episodes = int(N_EPISODES * EPSILON_DECAY)
    epsilon = EPSILON_START

    recent_rewards = [] # Rolling success rate
    rolling_log = []

    for episode in range(N_EPISODES):
        state, _ = env.reset(seed=seed + episode)
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

        recent_rewards.append(total_reward)

        if len(recent_rewards) > 500:
            recent_rewards.pop(0)

        # Decay epsilon linearly
        if episode < decay_episodes:
            epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * (episode / decay_episodes)
        else:
            epsilon = EPSILON_END

        # Save GIFs at beginning, middle, and end to see learned progression for only first seed
        if seed == 42 and episode in (1_000, 50_000, N_EPISODES - 1):
            save_gif(q_table, seed=seed, episode=episode, epsilon=epsilon)

        # Log every 500 episodes to W&B to show progression
        if episode % 500 == 0:
            rolling_log.append((episode, np.mean(recent_rewards)))
            print(f"Episode {episode}, Epsilon:{epsilon:.3f}, Reward: {total_reward}")
            wandb.log({"episode": episode, "reward": total_reward, "epsilon": epsilon, "rolling_success_rate": np.mean(recent_rewards)})

    print(f"Non-zero Q-table entries: {np.count_nonzero(q_table)}/{q_table.size}")

    env.close()

    success_rate = evaluate(q_table, seed=seed)
    wandb.log({"success_rate": success_rate})
    wandb.finish()
    return q_table, rolling_log


def evaluate(q_table: np.ndarray, seed: int = SEED, n_episodes: int = 1000) -> float:
    """Evaluate greedily over n episodes once training is complete to measure true learned performance (no ε)."""
    set_seed(seed)
    env = gym.make(ENV_ID, is_slippery=True, map_name=f"{MAP_SIZE}x{MAP_SIZE}")
    set_seed(seed, env)

    successes = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = int(np.argmax(q_table[state])) # Only greedy (no exploration)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        if reward == 1.0: # Goal is reached
            successes += 1

    env.close()
    success_rate = successes / n_episodes
    print(f"Success Rate: {success_rate:.1%} over {n_episodes} episodes")
    return success_rate

if __name__ == "__main__":
    q_table = train(seed=SEED)
    success_rate = evaluate(q_table, seed=SEED)