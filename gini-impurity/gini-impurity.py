import numpy as np

def compute_gini(y):
    if len(y)==0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    prob = counts/len(y)
    return 1 - np.sum(prob**2)

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right
    if n_total==0:
        return 0.0
    
    gini_left = compute_gini(y_left)
    gini_right = compute_gini(y_right)

    return (n_left/n_total)*gini_left + (n_right/n_total)*gini_right