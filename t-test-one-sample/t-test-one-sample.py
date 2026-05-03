import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    try:
        x = np.asarray(x, dtype=float)
        mu0 = float(mu0)
    except:
        return None
    
    n = len(x)

    mean = np.mean(x)
    s = np.std(x, ddof=1)

    t = (mean - mu0)*(np.sqrt(n))/s
    return float(t)