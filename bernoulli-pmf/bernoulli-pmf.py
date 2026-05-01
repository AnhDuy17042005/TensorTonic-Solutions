import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    try:
        x = np.asarray(x, dtype=float)
        p = float(p)
    except:
        raise ValueError("Cannot convert to numpy")

    # Compute PMF
    pmf = np.where(x==1, p, 1-p)
    # Compute mean
    mean = p
    # Compute variance
    var = p*(1-p)
    
    return (pmf, mean, var)