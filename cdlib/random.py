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
    previous_seed = _cdlib_global_seed
    seed(seed_value)

    # Monkey patching networkit algorithms if needed
    patched_classes = []
    original_inits = {}

    if nk is not None:
        try:
            from networkit.community import Infomap, PLM, Louvain, ParallelLeiden

            def patch_class(cls):
                original_init = cls.__init__

                def new_init(self, *args, **kwargs):
                    if 'seed' not in kwargs:
                        kwargs['seed'] = seed_value
                    original_init(self, *args, **kwargs)

                original_inits[cls] = original_init
                cls.__init__ = new_init
                patched_classes.append(cls)

            for cls in [Infomap, PLM, Louvain, ParallelLeiden]:
                patch_class(cls)

        except ImportError:
            pass  # Some classes may not exist depending on networkit version

    try:
        yield
    finally:
        if previous_seed is not None:
            seed(previous_seed)
        else:
            reset_seed()

        # Restore original networkit classes
        for cls in patched_classes:
            cls.__init__ = original_inits[cls]


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
    # Monkey patching networkit algorithms if needed
    patched_classes = []
    original_inits = {}
    if nk is not None:
        try:
            from networkit.community import Infomap, PLM, Louvain, ParallelLeiden

            def patch_class(cls):
                original_init = cls.__init__

                def new_init(self, *args, **kwargs):
                    if 'seed' not in kwargs:
                        kwargs['seed'] = seed_value
                    original_init(self, *args, **kwargs)

                original_inits[cls] = original_init
                cls.__init__ = new_init
                patched_classes.append(cls)

            for cls in [Infomap, PLM, Louvain, ParallelLeiden]:
                patch_class(cls)

        except ImportError:
            pass  # Some classes may not exist depending on networkit version


def get_seed(default=None):
    """Retrieve the global seed if set, else return a default value."""
    return _cdlib_global_seed if _cdlib_global_seed is not None else default


def reset_seed():
    """Reset the global seed to None (no forced seeding)."""
    global _cdlib_global_seed
    _cdlib_global_seed = None
