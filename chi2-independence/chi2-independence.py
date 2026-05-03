import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    try:
        C = np.asarray(C, dtype=float)
    except:
        return None
    
    total_col = np.sum(C, axis=0)
    total_row = np.sum(C, axis=1)
    total = np.sum(C)

    E = np.outer(total_row, total_col)/total    

    chi2 = np.sum((C - E)**2/E)

    return chi2, E