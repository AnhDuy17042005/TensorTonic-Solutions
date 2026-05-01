import numpy as np
from scipy.special import factorial

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    try:
        lam = float(lam)
        k = int(k)
    except:
        return None
    
    if lam <= 0 or k < 0:
        return None
    
    pmf, cdf = 0, 0

    # Compute PMF
    pmf = np.exp(-lam)*lam**k/factorial(k)

    # Compute CDF
    for i in range(k+1):
        cdf += (np.exp(-lam)*lam**i/factorial(i))

    return pmf, cdf