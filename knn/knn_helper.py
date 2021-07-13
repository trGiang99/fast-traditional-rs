import numpy as np
from numba import njit


@njit
def _predict(x_id, y_id, y_rated, S, k, k_min):
    """Predict rating of user x for item y (if iiCF).
    Args:
        x_id (int): users Id    (if iiCF)
        y_id (int): items Id    (if iiCF)
        y_rated (ndarray): List where element `i` is ndarray of `(xs, rating)` where `xs` is all x that rated y and the ratings.
        S (ndarray): similarity matrix
        k (int): number of k-nearest neighbors
        k_min (int): number of minimum k
    Returns:
        pred (float): predicted rating of user x for item y     (if iiCF)
    """

    k_neighbors = np.zeros((k, 2))
    k_neighbors[:, 1] = -1          # All similarity degree is default to -1

    for x2, rating in y_rated:
        if int(x2) == x_id:
            continue       # Bo qua item dang xet
        sim = S[int(x2), x_id]
        argmin = np.argmin(k_neighbors[:, 1])
        if sim > k_neighbors[argmin, 1]:
            k_neighbors[argmin] = np.array((sim, rating))

    # Compute weighted average
    sum_sim = sum_ratings = actual_k = 0
    for (sim, r) in k_neighbors:
        if sim > 0:
            sum_sim += sim
            sum_ratings += sim * r
            actual_k += 1

    if actual_k < k_min:
        sum_ratings = 0

    if sum_sim:
        est = sum_ratings / sum_sim

    return est


@njit
def _predict_mean(x_id, y_id, y_rated, mu, S, k, k_min):
    """Predict rating of user x for item y (if iiCF).
    Args:
        x_id (int): users Id    (if iiCF)
        y_id (int): items Id    (if iiCF)
        y_rated (ndarray): List where element `i` is ndarray of `(xs, rating)` where `xs` is all x that rated y and the ratings.
        mu (ndarray): List of mean ratings of all user (if iiCF, or all item if uuCF).
        S (ndarray): similarity matrix
        k (int): number of k-nearest neighbors
        k_min (int): number of minimum k
    Returns:
        pred (float): predicted rating of user x for item y     (if iiCF)
    """

    k_neighbors = np.zeros((k, 3))
    k_neighbors[:, 1] = -1          # All similarity degree is default to -1

    for x2, rating in y_rated:
        if int(x2) == x_id:
            continue       # Bo qua item dang xet
        sim = S[int(x2), x_id]
        argmin = np.argmin(k_neighbors[:, 1])
        if sim > k_neighbors[argmin, 1]:
            k_neighbors[argmin] = np.array((x2, sim, rating))

    # Compute weighted average
    sum_sim = sum_ratings = actual_k = 0
    for (nb, sim, r) in k_neighbors:
        nb = int(nb)
        if sim > 0:
            sum_sim += sim
            sum_ratings += sim * (r - mu[nb])
            actual_k += 1

    if actual_k < k_min:
        sum_ratings = 0

    if sum_sim:
        est = sum_ratings / sum_sim

    return est


@njit
def _predict_baseline(x_id, y_id, y_rated, S, k, k_min, global_mean, bx, by):
    """Predict rating of user x for item y (if iiCF) using baseline estimate.
    Args:
        x_id (int): users Id    (if iiCF)
        y_id (int): items Id    (if iiCF)
        y_rated (ndarray): List where element `i` is ndarray of `(xs, rating)` where `xs` is all x that rated y and the ratings.
        X (ndarray): the training set with size (|TRAINSET|, 3)
        S (ndarray): similarity matrix
        k (int): number of k-nearest neighbors
        k_min (int): number of minimum k
        global_mean (float): mean ratings in training set
        bx (ndarray): user biases   (if iiCF)
        by (ndarray): item biases   (if iiCF)
    Returns:
        pred (float): predicted rating of user x for item y     (if iiCF)
    """

    k_neighbors = np.zeros((k, 3))
    k_neighbors[:, 1] = -1          # All similarity degree is default to -1

    for x2, rating in y_rated:
        if int(x2) == x_id:
            continue       # Bo qua item dang xet
        sim = S[int(x2), x_id]
        argmin = np.argmin(k_neighbors[:, 1])
        if sim > k_neighbors[argmin, 1]:
            k_neighbors[argmin] = np.array((x2, sim, rating))

    est = global_mean + bx[x_id] + by[y_id]

    # Compute weighted average
    sum_sim = sum_ratings = actual_k = 0
    for (nb, sim, r) in k_neighbors:
        nb = int(nb)
        if sim > 0:
            sum_sim += sim
            nb_bsl = global_mean + bx[nb] + by[y_id]
            sum_ratings += sim * (r - nb_bsl)
            actual_k += 1

    if actual_k < k_min:
        sum_ratings = 0

    if sum_sim:
        est += sum_ratings / sum_sim

    return est
