# Project Journal
This journal logs the decisions made, problems encountered and my solutions to them, and the lessons learned throughout this project.

---

## Phase 0 Setup

### Environment
- Developed using Python 3.13 for support with ROCm 6.4, setup in a '.venv' virtual environment
- torch excluded from 'pyproject.toml' dependencies as needs to be manually installed for appropriate GPU
- ROCm only supports 3.11 - 3.13, so require this

### Tooling
- 'ruff' used for linting
- 'pytest' for testing
- Weights and Biases used for experiment tracking, to log reward curves, hyperparameters, and GIFS of trained agents
- CI runs 'ruff' and 'pytest' on every push

### Repo Structure
- 'algorithms/' : one file per algorithm (Q-learning, DQN, REINFORCE, PPO)
- 'common/' : shared utilities across all algorithms to prevent duplicate code
- 'configs/' : one YAML config per algorithm/environment pair
- 'results/' : store reward curves and GIFs
- 'tests/' : testing

Currently reading through 'Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto'

## Phase 1: Algorithm Implementation

### 1.1 Q-Learning
- FrozenLake-v1 chosen as the environment, on an 8x8 grid, with 'is_slippery' set to True.
- Q-learning will be the first algorithm implemented as it requires no neural network, focusing only on the core RL loop (state, action, reward, next state).
- The Bellman equation defines the optimal Q-table, where the Q-learning update rule moves towards it incrementally using TD error scaled by the learning rate alpha.
- ε-greedy exploration used, where ε decays linearly from 1.0 to 0.01 over 80% of training. This balances exploration early on, with exploitation later.
- A discount factor y = 0.99 chosen to value future rewards highly, which is useful for FrozenLake where the reward is only received at the end.
- Q-table initialised to zeros (64x4 for 8x8 FrozenLake with 4 actions)
- W&B configured to log seed and all hyperparameters per run, so that every result can be traced back to its exact configuration.
- ε-greedy policy for random action if ε > random number (explore), otherwise take action with the highest Q-value (exploitation).
- TD error zeros out on termination via (1 - terminated) to prevent future rewards
- ε decays linearly over 80% of training, then remains at 0.01
- Evaluation function uses greedy policy (no ε) over n episodes to measure true learned performance.