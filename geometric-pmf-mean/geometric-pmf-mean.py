import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    try:
        k = np.asarray(k, dtype=float)
        p = float(p)
    except:
        return None
    
    pmf = (1-p)**(k-1)*p
    mean = 1/p

    return pmf, float(mean)