import numpy as np

def gaussian_kernel(size, sigma):
    """
    Generate a normalized 2D Gaussian blur kernel.
    """
    G = np.zeros((size, size), dtype=float)
    center = size//2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            G[i][j] = np.exp(-(x**2 + y**2)/(2*sigma**2))
            
    G /= G.sum()
    return G.tolist()