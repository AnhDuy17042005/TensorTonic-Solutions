import numpy as np

def norm_of_vector(x: ndarray):
    norm_x = np.sqrt(np.sum(x**2))
    if norm_x == 0:
        return np.nan
    return norm_x
    
def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    try:
        v, w = np.asarray(v, dtype=float), np.asarray(w, dtype=float)
    except:
        return None

    # Compute magnitude
    norm_v = norm_of_vector(v)
    norm_w = norm_of_vector(w)

    # Compute denominator
    den = norm_v * norm_w
    den = np.where(den==0, 1, den)

    # Compute cos theta
    cos_theta = np.dot(v.T, w) / den
    cos_theta = np.clip(cos_theta, -1, 1)
    
    return np.arccos(cos_theta)