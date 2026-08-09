# Project Journal
This journal logs the decisions made, problems encountered and my solutions to them, and the lessons learned throughout this project.

---

### Environment
- Developed using Python 3.13 for support with ROCm 6.4, setup in a '.venv' virtual environment
- torch excluded from 'pyproject.toml' dependencies as needs to be manually installed for appropriate GPU
- ROCm only supports 3.11 - 3.13, so require this

### Tooling
- 'ruff' used for linting
- 'pytest' for testing
- Weights and Biases used for experiment tracking, to log reward curves, hyperparameters
- CI runs 'ruff' and 'pytest' on every push

### Repo Structure
- 'algorithms/' : one file per algorithm (Q-learning, DQN, REINFORCE, PPO)
- 'common/' : shared utilities across all algorithms to prevent duplicate code
- 'configs/' : one YAML config per algorithm/environment pair
- 'results/' : store reward curves and GIFs
- 'tests/' : testing

Currently reading through 'Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto'


## Q-Learning Implementation
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

### Results for Q-learning
- Achieved an average of approximately 59% success rate across multiple seeds and hyperparameter combinations on FrozenLake-V1 8x8 (is_slippery=True).
- Non-zero Q-table entries: 209/256 shows good state coverage confirming that exploration was not the limiting factor here.
- Performance plateau may be attributed to environment stochasticity due to is_slippery=True, (1/3 chance of sliding sideways).

### Possible Improvements
- Q-learning's convergence guarantee requires a decaying learning rate (Robbins-Monro conditions). A fixed decay rate here to be changed to visit-count based decay α = 1/(1 + N(s,a)).
- Further increase episodes due to the difficulty of 8x8 slippery FrozenLake.

### Visit-Count Learning Rate Second Attempt
- 1/(1+N(s,a)) decays too aggressively for bootstrapped TD learning, and performance worsened to ~24% success rate over 100,000 episodes.
- Frequently visited start states freeze within the first ~100 episodes while Q-values are noise, locking in bad estimates.
- To fix this, I have added a floor α = max(0.05, 1/(1+N(s,a))) to prevent premature freezing
- Performance significantly improved to a ~61% average success rate but no better than before a decaying learning rate was added. Due to this, I reverted back to a fixed alpha value.

### Learning Curve Analysis
- Added rolling success rate to track the learning progression
- All 5 seeds follow a near identical curve confirming reproducibility
- From episodes 0-40,000, we see the agent exploring randomly, with the goal rarely found (also seen in the video/gif). From episodes 40,000-80,000, we see a steady improvement as Q-values propagate from goal. Finally from episodes 80,000 to 100,000, we see a convergence at ~57%.

### Conclusion
- Despite numerous attempts at tuning hyperparameters, the ceiling for tabular Q-learning on this environment came at around ~62%. With is_slippery=True, the task became stochastic, with a 1/3 chance of sliding sideways, and so even the optimal action fails randomly, creating a hard ceiling.
- The learning curve shows us that the agent is steadily learning, with performance rising from near zero to ~57%, but plateaus well before the theoretical optimum.
- A fixed alpha value (α = 0.1) with 100k episodes performed equally or better than all other theoretically motivated alternatives.
- This motivates the use of function approximation in DQN, which can handle more complex environments.

## DQN Implementation
- CartPole-v1 chosen as the environment. It has a continuous state space (position, velocity, angle, angular velocity). A Q-table would not be able to represent the infinite possible states.
- DQN replaces the Q-table with a neural network that approximates the Q-values for any input state.
- This environment is a classic RL problem where the agent must balance a pole on a cart by only pushing left or right along a line, receiving +1 reward for every second the pole remains upright.
- The episode ends if: the pole angle exceeds 12 degrees, the cart moves out of bounds, or 500 steps are reached. Thus the maximum reward per episode and goal for this task is 500.
- There are two additions over Q-learning that make it more stable:
1. A replay buffer that stores past transitions and samples random batches to break correlation between consecutive updates.
2. A target network which is a frozen copy of the online network updated every N steps. This stabilises TD targets during training.

### Replay Buffer
- I implemented this as a fixed-size circular deque
- This stores transitions: (state, action, reward, next_state, terminated)
- When full, the oldest transitions are discarded automatically
- Random batch sampling breaks any temporal correlations between consecutive experiences

### Neural network
- Implemented a multi-layer perceptron for Q-value approximation
- It follows: 4 inputs -> 128 -> 128 -> 2 outputs, with ReLU activations between hidden layers, with no activation function on the output layer.
- The input dimension sizes is 4 (CartPole state: position, velocity, angle, angular velocity)
- The output dimension size is 2 (one Q-value per action: push left or push right)
- ReLU is chosen as the activation function due to its simplicity, and works well for value approximation tasks
- This is stored inside common/ to reuse in later algorithm files.

### Design Choices
- I have used two identical networks:
1. An online network which is updated every step via gradient descent
2. A target network that keeps a frozen copy, and is updated every 10 episodes
- The target network should stabilise TD targets during training. Without it, the target would change after every step and make learning unstable.
- I am using Adam optimiser as it is a standard choice for DQN, adapting the learning rate per parameter and is less sensitive to the learning rate hyperparameter, compared to other optimisation algorithms like Stochastic Gradient Descent (SGD)
- In DQN, the epsilon decay is multiplication, unlike in Q-learning where it was linear. This means it will decay faster earlier on, and slower later on, giving more exploitation time once the network has learned something useful.

### Predictions
- I believe after successful implementation, the agent should be able to consistently reach 450-500 reward per episode once trained. This is because CartPole is a much easier environment than FrozenLake 8x8 slippery.
- The learning curves will be more stable than in Q-learning.
- Faster convergence, with hopefully DQN solving this problem within 300 episodes.

### _update Function
- Takes a random batch of 64 transitions to sample from the replay buffer
- Loss computed as MSE between predicted Q-values and TD targets, backpropagated through the online network
- Future values zeroes on termination, like done in the Q-learning implementation

### Results on Initial DQN Run
- Seeds: [42, 50, 100, 1000, 1607]
- Mean rewards: ['195.2', '101.2', '500.0', '298.3', '422.8']
- Mean: 303.5, Std:  145.3

- My initial prediction was that DQN would consistently reach the 450-500 reward, but this now appears overconfident.
- Seed 100 achieved the maximum score of 500, and seed 1607 reaching 422.8. This shows that the algorithm can solve CartPole, but we also see that on seeds 42 and 50 perform significantly worse (195.2 and 101.2).
- With further research, this variance can be explained and is well documented in DQN due to:
1. With early random exploration differing per seed, the replay buffer fills with different quality experiences
2. Some seeds lead the agent into poor early paths that are difficult to recover from in just 500 episodes
- While this high variance came unexpected, it is not a sign the algorithm is broken, as seen in the higher reward runs. 
- Given these results, I will increase N_EPISODES to 1000 to allow for struggling seeds to allow the replay buffer to accumulate more diverse experiences. This reduces the negative impact an unlucky exploration may bring.

### Catastrophic Forgetting
- After increasing N_EPISODES to 1000, the final mean reward greatly dropped for seed 42. Up until episode 830, the agent learned well, with a rolling mean of 447.6, then suddenly performance collapsed and never recovered.
- This phenomenon is known in DQN and is called catastrophic forgetting, where the agent rapidly forgets previously learned information upon learning new information. 
- In this case, the agent was performing well with some exploration, until epsilon hit its minimum and the agent almost entirely stopped exploring. With a few bad experiences, the replay buffer got corrupted and the network started overwriting good Q-values with bad ones, thus getting stuck in a bad policy with no further exploration to recover.
- However, this was not the case for every seed, where for seed 50, the mean reward improved. This is expected as the collapse is sensitive to the specific sequence of experiences encountered during training.
- The results from increasing N_EPISODES to 1000 are as follows:
- Seeds: [42, 50, 100, 1000, 1607]
- Mean rewards: ['89.2', '167.4', '500.0', '261.9', '96.9']
- Mean: 223.1, Std:  151.7
- Seeds that collapsed early enough still had time to partially recover, as exploration continued, but for seed 42 that collapsed late (episode 830), this was not the case.


- I will try to minimise this from happening by increased the target update frequency from 10 to 20 for more stable TD targets, and add gradient clipping to prevent large updates from catastrophically overwriting learned weights. Also, by increasing the buffer capacity, the agent can learn from a more diverse set of experiences, and will hopefully reduce the chance of overwriting good policies with bad ones.

### Updated Results and Final Improvement
- For seed 42, this fix helped substantially, raising from a poor mean reward of 89.2 to a much better 345.4. However, this was not the case for all seeds and for seed 50, it dropped from 167.4 to 62.2. 
- This is a very poor performance, and so motivates the implementation of adding a 'best_net' that will save the best network during training and use that for evaluation, rather than the final network. This is equivalent to early stopping in supervised learning, where we use our peak learned performance rather than any instability that may have arisen in the final network.
- My first attempt at this used rolling training reward (epsilon-greedy), but this failed as early noisy spikes in a small window could permanently outrank later, more stable networks.
- I fixed this by replacing the rolling mean checkpointing with periodic greedy evaluation every 50 episodes (n_episodes = 5), to get a low variance, greedy signal.
- This increased the runtime, but greatly improved performance.

### Final DQN Results on CartPole-v1
#### Hyperparameters
- N_EPISODES = 1000 , BATCH_SIZE = 64, BUFFER_CAPACITY = 50_000
- GAMMA = 0.99, LR = 1e-3, EPSILON_START = 1.0, EPSILON_END = 0.01
- EPSILON_DECAY = 0.995 , TARGET_UPDATE_FREQ = 20
- Gradient clipping: max_norm = 10
- Best network selected via periodic greedy evaluation every 50 episodes

#### Results
- Seeds: [42, 50, 100, 1000, 1607]
- Mean rewards: 500.0, 500.0, 500.0, 500.0, 316.0
- Mean: 463.2, Std: 73.6
- 4 out of the 5 seeds achieved the maximum possible reward of 500

#### Analysis
- These final results are very strong, with 4/5 seeds achieving the maximum possible reward.
- Seed 1607 underperformed at 316.0, likely due to unlucky exploration early on.
- The standard deviation of 73.6 is a reflection of the outlier seed
- We cannot guarantee universal performance, but this consistent maximum reward across the first four seeds strongly suggested a learned policy.

#### Lessons Learned
- Vanilla DQN is prone to the phenomenon known as catastrophic forgetting when epsilon reaches its minimum, and the agent stops exploring with bad experiences corrupting the policy
- Three fixes were required to achieve the final performance:
1. A larger replay buffer, 10,000 to 50,000, allowed the agent to learn from a more diverse set of experiences, reducing the chance of overwriting good policies with bad ones.
2. Adding gradient clipping prevented large updates from overwriting learned weights. This was necessary as catastrophic forgetting was clearly evident in the rolling mean reward curves, where the agent peaked, and then suddenly collapsed.
3. Periodic greedy checkpointing every 50 episodes selected the best network rather than using noisy training rewards.
- This checkpoint fix appeared the most impactful, boosting an average score of ~200 up to 500 for most seeds. More specifically, our worst performing seed 42 from 195.2 to 500.

## Overall Conclusion
- Successfully implemented two foundational RL algorithms from scratch: tabular Q-learning and DQN, evaluated on FrozenLake-v1 and CartPole-v1 from OpenAI's Gym library, Gymnasium.
- Q-learning demonstrated the RL loop and Bellman equation, but hit a performance ceiling (~57%) due to the stochasticity of the environment.
- DQN showed that replacing the Q-table with a neural network, allowed the algorithm to handle continuous state space environments. In my case, achieving a mean reward of 463.2 out of a maximum 500 across 5 seeds.
- The most valuable lessons throughout came from debugging, especially understanding the results and the potential causes for any unexpected results. The problems I faced throughout this project were well known in the RL community and working through them helped consolidate my understand of these algorithms, beyond what theory alone provided.