import numpy as np
from numba import njit


@njit
def _predict(x_id, y_id, x_rated_y, S, k, k_min):
    """Predict rating of user x for item y (if iiCF).
    Args:
        x_id (int): users Id    (if iiCF)
        y_id (int): items Id    (if iiCF)
        x_rated_y (ndarray): List where element `i` is ndarray of `(xs, rating)` where `xs` is all x that rated y and the ratings.
        S (ndarray): similarity matrix
        k (int): number of k-nearest neighbors
        k_min (int): number of minimum k
    Returns:
        pred (float): predicted rating of user x for item y     (if iiCF)
    """

    k_neighbors = np.zeros((k, 2))
    k_neighbors[:, 1] = -1          # All similarity degree is default to -1

    for x2, rating in x_rated_y:
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
def _predict_mean(x_id, y_id, x_rated_y, mu, S, k, k_min):
    """Predict rating of user x for item y (if iiCF).
    Args:
        x_id (int): users Id    (if iiCF)
        y_id (int): items Id    (if iiCF)
        x_rated_y (ndarray): List where element `i` is ndarray of `(xs, rating)` where `xs` is all x that rated y and the ratings.
        mu (ndarray): List of mean ratings of all user (if iiCF, or all item if uuCF).
        S (ndarray): similarity matrix
        k (int): number of k-nearest neighbors
        k_min (int): number of minimum k
    Returns:
        pred (float): predicted rating of user x for item y     (if iiCF)
    """

    k_neighbors = np.zeros((k, 3))
    k_neighbors[:, 1] = -1          # All similarity degree is default to -1

    for x2, rating in x_rated_y:
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
def _predict_baseline(x_id, y_id, x_rated_y, S, k, k_min, global_mean, bx, by):
    """Predict rating of user x for item y (if iiCF) using baseline estimate.
    Args:
        x_id (int): users Id    (if iiCF)
        y_id (int): items Id    (if iiCF)
        x_rated_y (ndarray): List where element `i` is ndarray of `(xs, rating)` where `xs` is all x that rated y and the ratings.
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

    for x2, rating in x_rated_y:
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

@njit
def _calculate_precision_recall(user_ratings, k, threshold):
    """Calculate the precision and recall at k metric for the user based on his/her obversed rating and his/her predicted rating.

    Args:
        user_ratings (ndarray): An array contains the predicted rating in the first column and the obversed rating in the second column.
        k (int): the k metric.
        threshold (float): relevant threshold.

    Returns:
        (precision, recall): the precision and recall score for the user.
    """
    # Sort user ratings by estimated value
    user_ratings = user_ratings[user_ratings[:, 0].argsort()][::-1]

    # Number of relevant items
    n_rel = 0
    for _, true_r in user_ratings:
        if true_r >= threshold:
            n_rel += 1

    # Number of recommended items in top k
    n_rec_k = 0
    for est, _ in user_ratings[:k]:
        if est >= threshold:
            n_rec_k += 1

    # Number of relevant and recommended items in top k
    n_rel_and_rec_k = 0
    for (est, true_r) in user_ratings[:k]:
        if true_r >= threshold and est >= threshold:
            n_rel_and_rec_k += 1

    # Precision@K: Proportion of recommended items that are relevant
    # When n_rec_k is 0, Precision is undefined. We here set it to 0.
    if n_rec_k != 0:
        precision = n_rel_and_rec_k / n_rec_k
    else:
        precision = 0

    # Recall@K: Proportion of relevant items that are recommended
    # When n_rel is 0, Recall is undefined. We here set it to 0.
    if n_rel != 0:
        recall = n_rel_and_rec_k / n_rel
    else:
        recall = 0

    return precision, recall

@njit
def _calculate_ndcg(user_true_ratings, user_est_ratings, k):
    """Calculate the NDCG at k metric for the user based on his/her obversed rating and his/her predicted rating.

    Args:
        user_true_ratings (ndarray): An array contains the predicted rating on the test set.
        user_est_ratings (ndarray): An array contains the obversed rating on the test set.
        k (int): the k metric.

    Returns:
        ndcg: the ndcg score for the user.
    """
    # Sort user ratings by estimated value
    user_true_ratings_order = user_true_ratings.argsort()[::-1][:k]
    user_est_ratings_order = user_est_ratings.argsort()[::-1][:k]

    ndcg = dcg(user_est_ratings, user_est_ratings_order) / dcg(user_true_ratings, user_true_ratings_order)

    return ndcg

@njit
def dcg(ratings, order):
    """ Calculate discounted cumulative gain.

    Args:
        ratings (ndarray): the rating of the user on the test set.
        order (ndarray): list of item id, sorted by the rating.

    Returns:
        float: the discounted cumulative gain of the user.
    """
    dcg = 0
    for ith, item in enumerate(order):
        dcg += (np.power(2, ratings[item]) - 1) / np.log2(ith + 2)

    return dcg
