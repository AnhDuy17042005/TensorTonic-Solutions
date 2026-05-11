import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    tp = np.sum(y_true == y_pred)
    total = y_true.size

    fp = total - tp
    fn = total - tp

    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float((2 * tp) / denom)
