import numpy as np

def matrix_normalization(matrix, axis=None, norm_type="l2"):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """    
    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2:
        return None

    if norm_type not in ('l1', 'l2', 'max') or axis not in (0, 1, None):
        return None
    
    if norm_type == "l1":
        denominator = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        
    elif norm_type == "l2":
        denominator = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))
        
    elif norm_type == "max":
        denominator = np.max(np.abs(matrix), axis=axis, keepdims=True)

    denominator = np.where(denominator == 0, 1, denominator)

    return matrix / denominator