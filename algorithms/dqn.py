"""Deep Q-Network on CartPole-v1"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import wandb

from common.buffers import ReplayBuffer
from common.networks import MLP
from common.seed import set_seed

# Hyperparameters
ENV_ID = "CartPole-v1"
N_EPISODES = 500
BATCH_SIZE = 64
BUFFER_CAPACITY = 10_000
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
SEED = 1607

def _update(
        online_net: nn.Module,
        target_net: nn.Module,
        buffer: ReplayBuffer,
        optimiser: torch.optim.Optimizer,
        device: torch.device,
) -> float:
    """Sample a batch from the buffer and update the online network."""
    states, actions, rewards, next_states, terminateds = buffer.sample(BATCH_SIZE)

    # Convert to tensors for the nn's and move to GPU for computations
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    terminateds = torch.FloatTensor(terminateds).to(device)

    # Current Q-values from the online network
    # Unsqueeze(1) to reshape actions from (batch,) to (batch, 1) for gather
    # Use .gather to pick the Q-value for the action actually taken
    # squeeze(1) removes extra dimension, back to shape (batch,)
    current_q = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # TD targets from target network
    with torch.no_grad():
        next_q = target_net(next_states).max(1).values # Best Q-value across all actions in next state
        target_q = rewards + GAMMA * next_q * (1 - terminateds) # If episode ends, zero out future values

    # Compute loss and update
    loss = nn.functional.mse_loss(current_q, target_q) # The mean squared error between predicted and target Q-values
    optimiser.zero_grad()
    loss.backward() # Backpropagate
    optimiser.step() # Update network using computed gradients

    return loss.item()

def evaluate(net: nn.Module, seed: int = SEED, n_episodes: int = 10) -> float:
    """Evaluate the trained agent greedily over n episodes, returning mean reward."""
    device = torch.device("cpu")
    env = gym.make(ENV_ID)
    set_seed(seed, env)
    env.action_space.seed(seed)

    total_rewards = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=seed + i) # Different start per episode
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = net(state_tensor).argmax().item() # Take action with highest Q-value (greedy)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards.append(episode_reward)

    env.close()
    mean_reward = np.mean(total_rewards)
    print(f"Mean reward: {mean_reward:.1f} over {n_episodes} episodes")
    return float(mean_reward)


def train(seed: int = SEED) -> nn.Module:
    """Train DQN agent on CartPole."""
    set_seed(seed)

    env = gym.make(ENV_ID)
    env.action_space.seed(seed)
    set_seed(seed, env)

    state_dim = env.observation_space.shape[0] # For this env there are 4
    action_dim = env.action_space.n # For this env, 2

    device = torch.device("cpu")
    online_net = MLP(state_dim, action_dim).to(device)
    target_net = MLP(state_dim, action_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()  # Target network is never trained directly

    # Replay buffer and optimiser initialisation
    buffer = ReplayBuffer(BUFFER_CAPACITY)
    optimiser = torch.optim.Adam(online_net.parameters(), lr=LR) # Using Adam optimisation algorithm

    # Initialise W&B so every result can be traced back to its exact configuration
    wandb.init(
        project="rl-from-scratch",
        name=f"dqn-seed-{seed}",
        reinit="finish_previous",
        config={
            "algorithm": "DQN",
            "env": ENV_ID,
            "seed": seed,
            "batch_size": BATCH_SIZE,
            "buffer_capacity": BUFFER_CAPACITY,
            "gamma": GAMMA,
            "lr": LR,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "epsilon_decay": EPSILON_DECAY,
            "target_update_freq": TARGET_UPDATE_FREQ,
            "n_episodes": N_EPISODES,
        }
    )

    epsilon = EPSILON_START
    recent_rewards = []

    for episode in range(N_EPISODES):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = online_net(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            buffer.push(state, action, float(reward), next_state, terminated)

            state = next_state
            total_reward += reward

            # Train when the buffer has enough transitions
            if len(buffer) >= BATCH_SIZE:
                _update(online_net, target_net, buffer, optimiser, device)

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Every TARGET_UPDATE_FREQ episodes, update the target network
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Logging
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 50:
            recent_rewards.pop(0)

        if episode % 10 == 0:
            print(f"Episode {episode}, "
                  f"Reward: {total_reward}, "
                  f"Epsilon: {epsilon:.3f}, "
                  f"Rolling Mean: {np.mean(recent_rewards):.1f}")
            wandb.log({
                "episode": episode,
                "reward": total_reward,
                "epsilon": epsilon,
                "rolling_mean_reward": np.mean(recent_rewards)
            })

    env.close()
    mean_reward = evaluate(online_net, seed=seed)
    wandb.log({"eval_mean_reward": mean_reward})
    wandb.finish()
    return online_net

if __name__ == "__main__":
    net = train(seed=SEED)
    evaluate(net, seed=SEED)