import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    try:
        y_train = np.asarray(y_train, dtype=int)
        X_test  = np.asarray(X_test, dtype=int)
    except:
        return None
    
    if X_test.ndim > 0:
        n_test = X_test.shape[0]
    else:
        n_test = 1

    classes, counts = np.unique(y_train, return_counts=True)

    majority_cls = classes[np.argmax(counts)]
    
    return np.full(n_test, majority_cls)