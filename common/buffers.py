""""Replay buffer for DQN"""

from collections import deque
import random

import numpy as np


class ReplayBuffer:
    """Fixed-size circular buffer storing (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity) # automatically discard oldest experience when full

    def push(self, state, action, reward, next_state, terminated) -> None:
        """Store transition"""
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size: int) -> tuple:
        """Sample random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminateds = zip(*batch)
        return(
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(terminateds, dtype=np.float32),
    )

    def __len__(self) -> int:
        return len(self.buffer)
