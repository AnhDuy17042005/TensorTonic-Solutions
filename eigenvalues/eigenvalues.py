import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        matrix = np.asarray(matrix, dtype=float)
    except:
        return None
    
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    eigvals = np.linalg.eigvals(matrix)
    idx = np.lexsort((eigvals.imag, eigvals.real))

    return eigvals[idx]