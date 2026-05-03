import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    try:
        x = np.asarray(x, dtype=float)
        q = np.asarray(q, dtype=float)
    except:
        return None
    
    return np.percentile(x, q, method='linear')