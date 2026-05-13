import numpy as np

def histogram_equalize(image):
    """
    Apply histogram equalization to enhance image contrast.
    """
    image = np.asarray(image, dtype=int)

    values, counts = np.unique(image, return_counts=True)

    # Built histogram
    hist = np.zeros(256, dtype=int)
    hist[values]+=counts
    
    # Compute cdf
    cdf = np.cumsum(hist)

    total_pixels = image.size
    cdf_min = min(cdf[cdf > 0])

    if cdf_min == total_pixels:
        return np.zeros_like(image).tolist()
    
    # Built mapping
    mapping = np.round((cdf - cdf_min)/(total_pixels - cdf_min)*255).astype(int)

    return mapping[image].tolist()