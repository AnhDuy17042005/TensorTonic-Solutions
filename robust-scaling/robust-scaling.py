import numpy as np

def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    try:
        values = np.asarray(values, dtype=float)
    except:
        return None
    
    sorted_values = np.sort(values)
    n = len(values)

    if n == 0 or n == 1:
        return [0]
    
    median = np.median(sorted_values)
    mid = n // 2

    if n%2 == 0:
        lower = sorted_values[:mid]
        upper = sorted_values[mid:]
    else:
        lower = sorted_values[:mid]
        upper = sorted_values[mid+1:]

    q1, q3 = np.median(lower), np.median(upper)

    iqr = q3 - q1
    if iqr == 0:
        return (values - median).tolist()

    return ((values - median)/iqr).tolist()