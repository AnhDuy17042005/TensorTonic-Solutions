import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y, dtype=float)

    if len(y)==0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    prob = counts / len(y)

    return -np.sum(prob * np.log2(prob))