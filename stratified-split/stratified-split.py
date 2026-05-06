import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    try:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
    except:
        return None
    
    if X.shape[0]!= y.shape[0]:
        raise ValueError("X and y must have the same number of samples")  

    if not (0 < test_size < 1):
        raise ValueError("test size must be in (0, 1)")

    if rng is None:
        rng = np.random.default_rng()

    # Compute class and samples per class
    classes, counts = np.unique(y, return_counts=True)

    train_parts, test_parts = [], []

    for cls, count in zip(classes, counts):
        idx_cls = np.where(y == cls)[0]
        rng.shuffle(idx_cls)

        n_test = int(round(count * test_size))
        n_test = max(0, min(n_test, count))

        if count > 1:
            n_test = min(n_test, count - 1)

        test_idx = idx_cls[:n_test]
        train_idx = idx_cls[n_test:]

        test_parts.append(test_idx)
        train_parts.append(train_idx)

    train_indices = np.concatenate(train_parts)
    test_indices = np.concatenate(test_parts)

    train_indices = np.sort(train_indices)
    test_indices  = np.sort(test_indices)

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test