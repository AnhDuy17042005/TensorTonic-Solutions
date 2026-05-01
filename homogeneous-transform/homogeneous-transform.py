import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    try:
        points, T = np.asarray(points, dtype=float), np.asarray(T, dtype=float)
        points = np.atleast_2d(points)
    except:
        return None
    
    ones_col = np.ones((points.shape[0], 1))
    points = np.hstack((points, ones_col))

    new_points = (T @ points.T).T
    return np.squeeze(new_points[:, :-1])