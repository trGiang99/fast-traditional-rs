import numpy as np
from numba import njit


@njit
def _run_baseline_sgd(X, bx, by, global_mean, lr, reg):
    """Optimize biases using SGD.
    Args:
        X (ndarray): the training set with size (|TRAINSET|, 3)
        bx (ndarray): user specific biases (if uuCF)
        by (ndarray): item specific biases (if uuCF)
        global_mean (float): mean ratings in training set
        lr (float): the learning rate
        reg (float): the regularization strength
    Returns:
        A tuple ``(bx, by)``, which are users and items baselines.
    """

    for i in range(X.shape[0]):
        x, y, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]
        err = (rating - (global_mean + bx[x] + by[y]))
        bx[x] += lr * (err - reg * bx[x])
        by[y] += lr * (err - reg * by[y])

    return bx, by


@njit
def _run_baseline_als(bx, by, global_mean, n_x, n_y, x_rated, y_ratedby, reg_x, reg_y):
    """Optimize biases using ALS.
    Args:
        bx (ndarray): user specific biases (if uuCF)
        by (ndarray): item specific biases (if uuCF)
        global_mean (float): mean ratings in training set
        n_x (np.array): number of users
        n_y (np.array): number of items
        reg_x (float): regularization term for vector bx
        reg_y (float): regularization term for vector by
    Returns:
        A tuple ``(bx, by)``, which are users and items baselines.
    """

    for y in range(n_y):
        dev_y = 0
        for x, r in x_rated[y]:
            dev_y += r - global_mean - bx[int(x)]

        by[y] = dev_y / (reg_y + x_rated[y].shape[0])

    for x in range(n_x):
        dev_x = 0
        for y, r in y_ratedby[x]:
            dev_x += r - global_mean - by[int(y)]
        bx[x] = dev_x / (reg_x + y_ratedby[x].shape[0])

    return bx, by


@njit
def _compute_baseline_val_metrics(X_val, global_mean, bx, by):
    """Computes validation metrics (loss, rmse, and mae).
    Args:
        X_val (numpy array): validation set.
        global_mean (float): ratings arithmetic mean.
        bx (numpy array): users biases vector.
        by (numpy array): items biases vector.
    Returns:
        (tuple of floats): validation loss, rmse and mae.
    """
    residuals = np.zeros(X_val.shape[0])

    for i in range(X_val.shape[0]):
        x, y, rating = int(X_val[i, 0]), int(X_val[i, 1]), X_val[i, 2]
        pred = global_mean

        if x > -1:
            pred += bx[x]

        if y > -1:
            pred += by[y]

        residuals[i] = rating - pred

    loss = np.square(residuals).mean()
    rmse = np.sqrt(loss)
    mae = np.absolute(residuals).mean()

    return loss, rmse, mae
