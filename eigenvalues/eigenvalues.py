import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try: 
        matrix = np.asarray(matrix, dtype=float)    
    except (ValueError, TypeError):
        return None
    
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    
    eigenvalues = np.linalg.eigvals(matrix)
    idx = np.lexsort((eigenvalues.imag, eigenvalues.real))
    
    return eigenvalues[idx]