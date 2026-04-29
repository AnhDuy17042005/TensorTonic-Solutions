import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    den = 1
    try:
        matrix = np.asarray(matrix, dtype=float)
    except:
        None

    if matrix.ndim != 2:
        return None

    if norm_type not in ('l1', 'l2', 'max') or axis not in (0, 1, None):
        return None
        
    if norm_type == 'l1':
        den = np.sum(abs(matrix), axis=axis, keepdims=True)
    elif norm_type == 'l2':
        den = np.sqrt(np.sum(matrix**2, axis=axis, keepdims=True))
    elif norm_type == 'max':
        den = np.max(abs(matrix), axis=axis, keepdims=True)

    den = np.where(den == 0, 1, den)
    
    return matrix/den
        