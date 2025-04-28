# cdlib/utils/random.py

import random
import numpy as np
import os
from contextlib import contextmanager
import warnings

try:
    import igraph as ig
except ImportError:
    ig = None

try:
    import networkit as nk
except ImportError:
    nk = None

try:
    import sklearn
except ImportError:
    sklearn = None

try:
    import graph_tool as gt
except ImportError:
    gt = None

# Global variable to store the seed
_cdlib_global_seed = None


@contextmanager
def fixed_seed(seed_value: int):
    """Context manager to temporarily fix the seed."""
    previous_seed = _cdlib_global_seed  # no global needed here
    seed(seed_value)
    try:
        yield
    finally:
        if previous_seed is not None:
            seed(previous_seed)
        else:
            reset_seed()


def seed(seed_value: int):
    """Set a global random seed for reproducibility across cdlib and its dependencies."""
    global _cdlib_global_seed
    if _cdlib_global_seed is not None:
        warnings.warn(
            f"cdlib.seed() has already been set (previous value: {_cdlib_global_seed}). Overriding it.",
            UserWarning
        )
    _cdlib_global_seed = seed_value

    # Core Python
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    # Numpy
    np.random.seed(seed_value)

    # networkit
    if nk is not None:
        nk.engine.setSeed(seed_value, False)


def get_seed(default=None):
    """Retrieve the global seed if set, else return a default value."""
    return _cdlib_global_seed if _cdlib_global_seed is not None else default


def reset_seed():
    """Reset the global seed to None (no forced seeding)."""
    global _cdlib_global_seed
    _cdlib_global_seed = None
