import numpy as np

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample binary log loss.
    """
    
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)

    y_pred = np.clip(y_pred, eps, 1 - eps)

    losses = []

    for y, p in zip(y_true, y_pred):

        # If true label is 1
        if y == 1:
            loss = -np.log(p)

        # If true label is 0
        else:
            loss = -np.log(1 - p)

        losses.append(float(loss))

    return losses