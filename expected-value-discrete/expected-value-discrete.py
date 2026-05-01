import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    try:
        x, p = np.asarray(x, dtype=float), np.asarray(p, dtype=float)
    except:
        return None

    if x.shape != p.shape:
        raise ValueError("Shape mismatch")

    if np.allclose(np.sum(p), 1.0) == False:
        raise ValueError("Probabilities must sum to 1")

    return x @ p