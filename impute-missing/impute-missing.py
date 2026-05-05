def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    try:
        X = np.asarray(X, dtype=float)
    except:
        return None
    
    if strategy not in ["mean", "median"]:
        return None
    
    # Find mean or median
    if strategy == "mean":
        method = np.nanmean(X, axis=0)
    else:
        method = np.nanmedian(X, axis=0)

    # Find NaN values
    mask = np.logical_not(np.isnan(X))

    # Handle all-NaN columns
    valid_cols = np.any(mask, axis=0)
    method = np.where(valid_cols==False, 0, method)

    output = np.where(mask==True, X, method)
    return output