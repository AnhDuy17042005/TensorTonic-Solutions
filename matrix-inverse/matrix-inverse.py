import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    A = np.asarray(A, dtype=float)
    det = np.linalg.det(A)

    if A.shape[0] != A.shape[1] or det == 0:
        return None
    
    return np.linalg.inv(A)
