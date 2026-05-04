import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    try:
        X = np.asarray(X, dtype=float)
    except:
        return None
    
    max_x = np.max(X, axis=axis, keepdims=True)
    min_x = np.min(X, axis=axis, keepdims=True)

    den = (max_x - min_x)
    den = np.maximum(den, eps)
    
    output = (X - min_x)/den
    return output