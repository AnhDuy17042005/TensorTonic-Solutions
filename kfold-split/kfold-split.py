import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    arr = np.arange(N)

    if shuffle:
        if rng is not None:
            arr = rng.permutation(arr)
        else:
            np.random.shuffle(arr)

    folds = np.array_split(arr, k)

    results = []

    for i in range(k):
        val = folds[i]

        train = np.concatenate([
            folds[j] for j in range(k) if j!=i
            ])
        results.append((train, val))

    return results