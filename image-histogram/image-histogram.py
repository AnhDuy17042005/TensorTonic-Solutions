import numpy as np

def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    image = np.asarray(image, dtype=int)

    values, counts = np.unique(image, return_counts=True)
    histogram = np.zeros(256)

    histogram[values]+=counts
    
    return histogram.tolist()