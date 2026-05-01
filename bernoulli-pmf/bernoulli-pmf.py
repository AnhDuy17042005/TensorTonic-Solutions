import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    try:
        x = np.asarray(x, dtype=float)
    except:
        raise ValueError("Cannot convert to numpy")

    # Compute PMF
    pmf = p**x*(1 - p)**(1-x)
    # Compute mean
    mean = p
    # Compute variance
    var = p*(1-p)
    
    return (pmf, mean, var)