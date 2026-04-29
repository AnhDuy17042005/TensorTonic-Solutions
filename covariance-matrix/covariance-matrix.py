import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    try:
        X = np.asarray(X, dtype=float)
    except:
        return None
        
    if X.ndim != 2 or X.shape[0] <= 1:
        return None
    N, D = X.shape[:2]
        
    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    return (1/(N-1) * X_centered.T @ X_centered)    