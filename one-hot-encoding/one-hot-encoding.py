import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    
    try:
        y = np.asarray(y, dtype=int)
    except:
        return None
    
    if num_classes == None:
        num_classes = int(max(y) + 1)

    N = len(y)

    Y = np.zeros((N, num_classes), dtype=int)
    Y[np.arange(len(y)), y] = 1
    
    return Y