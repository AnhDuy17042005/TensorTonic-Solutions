import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    n = int(n)
    p = float(p)
    k = int(k)

    pmf, cdf = 0, 0
    
    pmf = comb(n, k)*(p**k)*(1-p)**(n-k)
    for i in range(k+1):
        cdf += comb(n, i)*(p**i)*(1-p)**(n-i)

    return float(pmf), float(cdf)