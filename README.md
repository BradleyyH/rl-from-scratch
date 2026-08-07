# Reinforcement Learning From Scratch

Tabular Q-learning and DQN implemented from first principles in PyTorch, evaluated on Gymnasium benchmarks with seeded runs, W&B experiment tracking, and a techinical journal documenting design decisions and lessons learned.

## Results

| Algorithm | Environment | Mean Return (5 seeds) |
|-----------|-------------|----------------------|
| Q-learning | FrozenLake-v1 | TBD |
| DQN | CartPole-v1 | TBD |

## Algorithms

### Planned

- [ ] **Tabular Q-learning** - FrozenLake-v1
- [ ] **DQN** - CartPole-v1

## Installation

1. Install PyTorch for either AMD or Nvidia GPUs:
- ROCm (AMD):   pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
- CUDA (Nvidia): pip install torch
- CPU only:      pip install torch
2. pip install -e ".[dev]"

## Usage

## Experiment Tracking

Runs will be logged to Weights & Biases, with each run stating its seed.

## Journal
The journal here logs the decisions made, problems encountered; solutions to them, and the lessons learned throughout this project: [JOURNAL.md](JOURNAL.md)