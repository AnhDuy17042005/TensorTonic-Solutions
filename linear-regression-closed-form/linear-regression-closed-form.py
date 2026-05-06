import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    try:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype = float)
    except:
        return None
    
    return np.linalg.inv(X.T @ X) @ X.T @ y