# Reinforcement Learning From Scratch

Tabular Q-learning and DQN implemented from first principles in PyTorch, evaluated on Gymnasium benchmarks with seeded runs, W&B experiment tracking, and a technical journal documenting design decisions and lessons learned.

## Results

| Algorithm | Environment | Mean Return (5 seeds) | Std |
|-----------|-------------|--------------| ----- |
| Q-learning | FrozenLake-v1 | 57.3%        | ±5.6% |
| DQN | CartPole-v1 | 463.2 / 500  | ±73.6 |

### Q-learning — FrozenLake-v1

<table>
  <tr>
    <td><img src="results/q_learning_success_rates.png" width="400"/></td>
    <td><img src="results/q_learning_heatmap.png" width="400"/></td>
  </tr>
</table>
<td><img src="results/q_learning_learning_curve.png" width="600"/></td>

#### Learning Progression
| Early (Episode 1,000) | Mid (Episode 50,000) | Final (Episode 99,999) |
|----------------------|---------------------|----------------------|
| ![Early](results/q_learning_episode_1000_seed_42.gif) | ![Mid](results/q_learning_episode_50000_seed_42.gif) | ![Final](results/q_learning_episode_99999_seed_42.gif) |

### DQN — CartPole-v1

<table>
  <tr>
    <td align="center"><img src="results/dqn_mean_rewards.png" width="400"/></td>
    <td align="center"><img src="results/dqn_learning_curves.png" width="400"/></td>
  </tr>
</table>

#### Learning Progression
| Early (Episode 50)                           | Mid (Episode 500) | Best Policy Learned                            |
|----------------------------------------------|------------------|------------------------------------------------|
| ![Early](results/dqn_episode_50_seed_42.gif) | ![Mid](results/dqn_episode_500_seed_42.gif) | ![Final](results/dqn_episode_best_seed_42.gif) |
 > Click the gifs to see the algorithm in action

## Installation

1. `pip install torch`
2. `pip install -e ".[dev]"`

> DQN currently runs using CPU. GPU support depends on hardware and ROCm/CUDA compatibility. As the environments here are low complexity, the CPU is sufficient.

## Usage
```bash
python algorithms/q_learning.py     # Train Q-learning on FrozenLake environment
python algorithms/dqn.py            # Train DQN on CartPole environment
python scripts/run_q_learning.py    # Run Q-learning over 5 seeds
python scripts/run_dqn.py           # Run DQN over 5 seeds
```

## Experiment Tracking

All runs are logged to Weights & Biases, with each run stating its seed and full hyperparameter configuration so every result can be fully reproduced.

## Journal
The journal here logs the decisions made, problems encountered; solutions to them, and the lessons learned throughout this project: [JOURNAL.md](JOURNAL.md)