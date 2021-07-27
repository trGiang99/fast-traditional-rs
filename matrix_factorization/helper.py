import numpy as np
from numba import njit


@njit
def _shuffle(X):
    np.random.shuffle(X)
    return X

@njit
def _run_svd_epoch(X, pu, qi, bu, bi, global_mean, n_factors, lr_pu, lr_qi, lr_bu, lr_bi, reg_pu, reg_qi, reg_bu, reg_bi):
    """Runs an SVD epoch, updating model weights (pu, qi, bu, bi).
    Args:
        X (numpy array): training set.
        pu (numpy array): users latent factor matrix.
        qi (numpy array): items latent factor matrix.
        bu (numpy array): users biases vector.
        bi (numpy array): items biases vector.
        global_mean (float): ratings arithmetic mean.
        n_factors (int): number of latent factors.
        lr_pu (float): the learning rate for Pu.
        lr_qi (float): the learning rate for Qi.
        lr_bu (float): the learning rate for bu.
        lr_bi (float): the learning rate for bi.
        reg_pu (float): regularization factor for Pu.
        reg_qi (float): regularization factor for Qi.
        reg_bu (float): regularization factor for bu.
        reg_bi (float): regularization factor for bi.
    Returns:
        pu (numpy array): users latent factor matrix updated.
        qi (numpy array): items latent factor matrix updated.
        bu (numpy array): users biases vector updated.
        bi (numpy array): items biases vector updated.
        train_loss (float): training loss.
    """
    residuals = []
    for i in range(X.shape[0]):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]

        # Predict current rating
        pred = global_mean + bu[user] + bi[item]

        for factor in range(n_factors):
            pred += pu[user, factor] * qi[item, factor]

        err = rating - pred
        residuals.append(err)

        # Update biases
        bu[user] += lr_bu * (err - reg_bu * bu[user])
        bi[item] += lr_bi * (err - reg_bi * bi[item])

        # Update latent factors
        for factor in range(n_factors):
            puf = pu[user, factor]
            qif = qi[item, factor]

            pu[user, factor] += lr_pu * (err * qif - reg_pu * puf)
            qi[item, factor] += lr_qi * (err * puf - reg_qi * qif)

    residuals = np.array(residuals)
    train_loss = np.square(residuals).mean()
    return pu, qi, bu, bi, train_loss


@njit
def _compute_svd_val_metrics(X_val, pu, qi, bu, bi, global_mean, n_factors):
    """Computes validation metrics (loss, rmse, and mae) for SVD.
    Args:
        X_val (numpy array): validation set.
        pu (numpy array): users latent factor matrix.
        qi (numpy array): items latent factor matrix.
        bu (numpy array): users biases vector.
        bi (numpy array): items biases vector.
        global_mean (float): ratings arithmetic mean.
        n_factors (int): number of latent factors.
    Returns:
        (tuple of floats): validation loss, rmse and mae.
    """
    residuals = []

    for i in range(X_val.shape[0]):
        user, item, rating = int(X_val[i, 0]), int(X_val[i, 1]), X_val[i, 2]
        pred = global_mean

        if user > -1:
            pred += bu[user]

        if item > -1:
            pred += bi[item]

        if (user > -1) and (item > -1):
            for factor in range(n_factors):
                pred += pu[user, factor] * qi[item, factor]

        residuals.append(rating - pred)

    residuals = np.array(residuals)
    loss = np.square(residuals).mean()
    rmse = np.sqrt(loss)
    mae = np.absolute(residuals).mean()

    return loss, rmse, mae


@njit
def _run_svdpp_epoch(X, pu, qi, bu, bi, yj, global_mean, n_factors, I, lr_pu, lr_qi, lr_bu, lr_bi, lr_yj, reg_pu, reg_qi, reg_bu, reg_bi, reg_yj):
    """Runs an SVD++ epoch, updating model weights (pu, qi, bu, bi).
    Args:
        X (numpy array): training set.
        pu (numpy array): users latent factor matrix.
        qi (numpy array): items latent factor matrix.
        bu (numpy array): users biases vector.
        bi (numpy array): items biases vector.
        bi (numpy array): items biases vector.
        yj (numpy array): The implicit item factors.
        global_mean (float): ratings arithmetic mean.
        n_factors (int): number of latent factors.
        lr_pu (float): the learning rate for Pu.
        lr_qi (float): the learning rate for Qi.
        lr_bu (float): the learning rate for bu.
        lr_bi (float): the learning rate for bi.
        lr_yj (float): the learning rate for yj.
        reg_pu (float): regularization factor for Pu.
        reg_qi (float): regularization factor for Qi.
        reg_bu (float): regularization factor for bu.
        reg_bi (float): regularization factor for bi.
        reg_yj (float): regularization factor for yj.
    Returns:
        pu (numpy array): users latent factor matrix updated.
        qi (numpy array): items latent factor matrix updated.
        bu (numpy array): users biases vector updated.
        bi (numpy array): items biases vector updated.
        train_loss (float): training loss.
    """
    residuals = []
    for i in range(X.shape[0]):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]

        # Items rated by user u
        last_ele = np.where(I[user] == -1)[0][0]
        Iu = I[user, :last_ele]

        # Square root of number of items rated by user u
        sqrt_Iu = np.sqrt(Iu.shape[0])

        # Compute user implicit feedback
        u_impl_fdb = np.zeros(n_factors, np.double)
        for j in Iu:
            for factor in range(n_factors):
                u_impl_fdb[factor] += yj[j, factor] / sqrt_Iu

        # Predict current rating
        pred = global_mean + bu[user] + bi[item]

        for factor in range(n_factors):
            pred += qi[item, factor] * (pu[user, factor] + u_impl_fdb[factor])

        err = rating - pred
        residuals.append(err)

        # Update biases
        bu[user] += lr_bu * (err - reg_bu * bu[user])
        bi[item] += lr_bi * (err - reg_bi * bi[item])

        # Update latent factors
        for factor in range(n_factors):
            puf = pu[user, factor]
            qif = qi[item, factor]

            pu[user, factor] += lr_pu * (err * qif - reg_pu * puf)
            qi[item, factor] += lr_qi * (err * (puf + u_impl_fdb[factor]) - reg_qi * qif)

            for j in Iu:
                yj[j, factor] += lr_yj * (err * qif / sqrt_Iu - reg_yj * yj[j, factor])

    residuals = np.array(residuals)
    train_loss = np.square(residuals).mean()
    return pu, qi, bu, bi, yj, train_loss

@njit
def _compute_svdpp_val_metrics(X_val, pu, qi, bu, bi, yj, global_mean, n_factors, I):
    """Computes validation metrics (loss, rmse, and mae) for SVD++.
    Args:
        X_val (numpy array): validation set.
        pu (numpy array): users latent factor matrix.
        qi (numpy array): items latent factor matrix.
        bu (numpy array): users biases vector.
        bi (numpy array): items biases vector.
        yj (numpy array): The implicit item factors.
        global_mean (float): ratings arithmetic mean.
        n_factors (int): number of latent factors.
    Returns:
        (tuple of floats): validation loss, rmse and mae.
    """
    residuals = []

    for i in range(X_val.shape[0]):
        user, item, rating = int(X_val[i, 0]), int(X_val[i, 1]), X_val[i, 2]

        # Items rated by user u
        Iu = I[user]
        last_ele = np.where(Iu == -1)[0][0]
        Iu = Iu[:last_ele]

        # Square root of number of items rated by user u
        sqrt_Iu = np.sqrt(Iu.shape[0])

        pred = global_mean

        u_impl_fdb = np.zeros(n_factors, np.double)
        for j in Iu:
            for factor in range(n_factors):
                u_impl_fdb[factor] += yj[j, factor] / sqrt_Iu

        if user > -1:
            pred += bu[user]

        if item > -1:
            pred += bi[item]

        if (user > -1) and (item > -1):
            for factor in range(n_factors):
                pred += qi[item, factor] * (pu[user, factor] + u_impl_fdb[factor])

        residuals.append(rating - pred)

    residuals = np.array(residuals)
    loss = np.square(residuals).mean()
    rmse = np.sqrt(loss)
    mae = np.absolute(residuals).mean()

    return loss, rmse, mae
