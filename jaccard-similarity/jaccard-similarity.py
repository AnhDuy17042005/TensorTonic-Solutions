def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    set_a = set(set_a)
    set_b = set(set_b)

    intersection = set_a & set_b
    union = set_a | set_b

    if len(union) == 0:
        return 0

    return len(intersection)/len(union)