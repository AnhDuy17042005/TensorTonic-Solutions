import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    try:
        x = np.asarray(x, dtype=float)
    except:
        return None
    
    mean = np.mean(x)
    n = len(x)

    var = np.sum(1/(n-1)*(x-mean)**2)
    std = np.std(x, ddof = 1)

    return var, std