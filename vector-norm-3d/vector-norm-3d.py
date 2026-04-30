import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    try:
        v = np.asarray(v, dtype=float)
    except:
        return None

    return np.sqrt(np.sum(v**2, axis=-1))