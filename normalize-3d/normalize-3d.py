import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    try:
        v = np.asarray(v, dtype=float)
    except:
        return None

    den = np.sqrt(np.sum(v**2, axis=-1, keepdims=True))
    den = np.where(den==0, 1, den)
    
    return v/den