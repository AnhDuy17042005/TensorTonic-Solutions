import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    try:
        x = np.asarray(x, dtype=float)
    except:
        return None
    
    # Compute mean
    mean = np.mean(x)

    # Compute median
    median = np.median(x)

    # Compute mode
    counter = Counter(x)
    max_freq = max(counter.values())

    for k, v in counter.items():
        if v == max_freq:
            mode = k
            break

    return mean, median, mode