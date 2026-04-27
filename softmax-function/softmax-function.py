import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.asarray(x, dtype=float)
    num = np.exp(x-np.max(x))
    den = np.sum(np.exp(x-np.max(x)), axis=-1, keepdims=True)
    return num/den