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
