import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    If k > number of training samples, pad with -1.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    k = int(k)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    distance = np.sqrt(np.sum(diff ** 2, axis=2))

    indices = np.argsort(distance, axis=1)

    n_test = X_test.shape[0]
    n_train = X_train.shape[0]

    result = np.full((n_test, k), -1, dtype=int)

    take = min(k, n_train)
    result[:, :take] = indices[:, :take]

    return result
