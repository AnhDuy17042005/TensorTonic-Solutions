import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    A = np.asarray(A, dtype=float)
    determinant = np.linalg.det(A)

    if A.ndim != 2:
        return None
    
    if determinant == 0:
        return None

    return np.linalg.inv(A)
    