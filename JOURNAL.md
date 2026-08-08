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

## Results for Q-learning
- Achieved an average of approximately 59% success rate across multiple seeds and hyperparameter combinations on FrozenLake-V1 8x8 (is_slippery=True).
- Non-zero Q-table entries: 209/256 shows good state coverage confirming that exploration was not the limiting factor here.
- Performance plateau may be attributed to environment stochasticity due to is_slippery=True, (1/3 chance of sliding sideways).

## Possible Improvements
- Q-learning's convergence guarantee requires a decaying learning rate (Robbins-Monro conditions). A fixed decay rate here to be chnaged to visit-count based decay α = 1/(1 + N(s,a)).
- Further increase episodes due to the difficulty of 8x8 slippery FrozenLake.

## Visit-Count Learning Rate Second Attempt
- 1/(1+N(s,a)) decays too aggressively for bootstrapped TD learning, and performance worsened to ~24% success rate over 100,000 episodes.
- Frequently visited start states freeze within the first ~100 episodes while Q-values are noise, locking in bad estimates.
- To fix this, I have added a floor α = max(0.05, 1/(1+N(s,a))) to prevent premature freezing
- Performance significantly improved to a ~61% average success rate but no better than before a decaying learning rate was added. Due to this, I reverted back to a fixed alpha value.

## Learning Curve Analysis
- Added rolling success rate to track the learning progression
- All 5 seeds follow a near identical curve confirming reproducibility
- From episodes 0-40,000, we see the agent exploring randomly, with the goal rarely found (also seen in the video/gif). From episodes 40,000-80,000, we see a steady improvement as Q-values propagate from goal. Finally from episodes 80,000 to 100,000, we see a convergence at ~57%.

## Conclusion
- Despite numerous attempts at tuning hyperparameters, the ceiling for tabular Q-learning on this environment came at around ~62%. With is_slippery=True, the task became stochatic, with a 1/3 chance of sliding sideways, and so even the optimal action fails randomly, creating a hard ceiling.
- The learning curve shows us that the agent is steadily learning, with performance rising from near zero to ~57%, but platuaus well before the theoretical optimum.
- A fixed alpha value (α = 0.1) with 100k episodes performed equally or better than all other theoretically motivated alternatives.
- This motivates the use of function approximation in DQN, which can handle more complex environments.
