"""
Reproducibility helper. Call set_seed() once at the start of any
script that involves randomness (data synthesis, splitting, training).
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42, deterministic_torch: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Torch is optional at Phase 1 time; seed it only if installed/used later.
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
