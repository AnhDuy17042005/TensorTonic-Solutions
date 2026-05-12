import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    classes = np.unique(y_train)
    n_samples, n_features = X_train.shape

    results = []

    for x in X_test:
        class_scores = []

        for c in classes:
            X_c = X_train[y_train == c]
            n_c = len(X_c)

            log_prior = np.log(n_c / n_samples)

            theta = (np.sum(X_c, axis=0) + 1) / (n_c + 2)

            # Bernoulli log likelihood
            log_likelihood = np.sum(
                x * np.log(theta) + (1 - x) * np.log(1 - theta)
            )

            scorce = log_prior + log_likelihood
            class_scores.append(scorce)
        
        results.append(class_scores)

    return np.asarray(results)