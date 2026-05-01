import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    try:
        n = int(n)
        p = float(p)
        k = int(k)
    except:
        return None

    if n < 0 or p < 0 or p > 1 or k < 0 or k > n:
        return None
    
    pmf, cdf = 0, 0
    
    pmf = comb(n, k)*(p**k)*(1-p)**(n-k)
    for i in range(k+1):
        cdf += comb(n, i)*(p**i)*(1-p)**(n-i)

    return float(pmf), float(cdf)