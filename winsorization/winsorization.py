import numpy as np

def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    """
    try:
        values = np.asarray(values, dtype=int)
    except:
        return None

    lower = np.percentile(values, lower_pct)
    upper = np.percentile(values, upper_pct)

    return np.clip(values, lower, upper).tolist()