import numpy as np

def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    """
    try:
        values = np.asarray(values, dtype=int)
    except:
        return None
    
    n = len(values)

    lower = np.percentile(values, lower_pct)
    upper = np.percentile(values, upper_pct)

    values = np.where(values<lower, lower, values)
    values = np.where(values>upper, upper, values)

    print(values)

    return values.tolist()