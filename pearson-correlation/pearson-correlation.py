import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    try:
        X = np.asarray(X, dtype=float)
    except:
        return None

    if X.ndim != 2 or X.shape[0] <= 1:
        return None
    # Samples and features
    N, D = X.shape[:2]

    # Centered data
    mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - mean
    # Covariance matrix
    covariance_matrix = 1/(N - 1) * X_centered.T @ X_centered

    # Standard deviation
    std = np.std(X_centered, axis=0, ddof=1)
    # Denominator matrix
    den = np.outer(std, std)
    
    # Corralation matrix
    correlation_matrix = covariance_matrix/den
    
    # Handle zero variance features
    correlation_matrix = np.where(den == 0, np.nan, correlation_matrix)
        
    return correlation_matrix