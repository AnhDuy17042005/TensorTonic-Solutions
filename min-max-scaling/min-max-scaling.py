import numpy as np

def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    try:
        data = np.asarray(data, dtype=float)
    except:
        return None
    
    min_data = np.min(data, axis=0, keepdims=True)
    max_data = np.max(data, axis=0, keepdims=True)

    den = max_data - min_data
    den = np.where(den == 0, 1, den)

    return ((data - min_data)/den).tolist()