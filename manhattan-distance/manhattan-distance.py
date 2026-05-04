import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    try:
        x = np.asarray(x, dtype=float)
    except:
        return None
    
    return np.sum(np.abs(x - y))