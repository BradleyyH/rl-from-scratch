"""Seeding utility to be called at the start of each run"""

import random

import numpy as np
import torch


def set_seed(seed: int, env=None) -> None:
    """Seed Python, NumPy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Covers multiple GPUs if using multiple

    # If Gymnasium environment is used, reset it with the same seed
    if env is not None:
        env.reset(seed=seed)
