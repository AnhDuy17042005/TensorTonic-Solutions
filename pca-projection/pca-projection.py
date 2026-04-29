import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    try:
        X = np.asarray(X, dtype=float)
    except:
        return None

    N, D = X.shape[:2]
    if X.ndim != 2 or N < 2 or k > D:
        return None

    # Centered data
    mean = np.mean(X, axis=0, keepdims=True)
    X_c = X - mean

    # Covariance matrix
    cov = 1/(N - 1) * X_c.T @ X_c

    # Compure eigenvalues, eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort values
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Top k
    W = eigenvectors[:, :k]

    return X_c @ W