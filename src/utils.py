"""Shared reproducibility helpers for Project 28 (no-reference IQA).

Course rule: a fixed random seed must produce every headline result.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed the Python, NumPy and PyTorch (CPU + CUDA) RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
