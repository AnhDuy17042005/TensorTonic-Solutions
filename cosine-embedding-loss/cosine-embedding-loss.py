import numpy as np

def norm_vector(x):
    return np.sqrt(np.sum(x**2))

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)

    norm_x1 = norm_vector(x1)
    norm_x2 = norm_vector(x2)

    cos = (x1 @ x2)/(norm_x1 * norm_x2)

    if label == 1:
        L = 1 - cos
    if label == -1:
        L = np.maximum(0, cos - margin)

    return float(L)
