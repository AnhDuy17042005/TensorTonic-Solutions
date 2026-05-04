import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if rng is None:
        rng = np.random.default_rng()

    boot_means = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = x[idx]
        boot_means[b] = np.mean(sample)

    alpha = 1 - ci
    lower = np.quantile(boot_means, alpha/2)
    upper = np.quantile(boot_means, 1-alpha/2)

    return boot_means, lower, upper