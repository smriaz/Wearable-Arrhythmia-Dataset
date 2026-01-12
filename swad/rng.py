import numpy as np

def make_rng(seed: int) -> np.random.Generator:
    """Create a deterministic NumPy RNG."""
    return np.random.default_rng(int(seed))
