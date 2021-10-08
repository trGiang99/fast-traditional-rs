import numpy as np
import pandas as pd
from math import sqrt
from scipy import sparse
import time
import pickle

from utils import timer
from .helper import _run_svdpp_epoch, _compute_svdpp_val_metrics, _shuffle
from .svd import SVD

class SVDpp(SVD):
    def __init__(self, learning_rate=.005, lr_pu=None, lr_qi=None, lr_bu=None, lr_bi=None, lr_yj=None,
                 regularization=0.02, reg_pu=None, reg_qi=None, reg_bu=None, reg_bi=None, reg_yj=None,
                 n_epochs=20, n_factors=100, min_rating=1, max_rating=5, i_factor=None, u_factor=None):
        SVD.__init__(self, learning_rate, lr_pu, lr_qi, lr_bu, lr_bi,
                 regularization, reg_pu, reg_qi, reg_bu, reg_bi,
                 n_epochs, n_factors, min_rating, max_rating, i_factor, u_factor)
        self.lr_yj = lr_yj if lr_yj is not None else learning_rate
        self.reg_yj = reg_yj if reg_yj is not None else regularization

    def _sgd(self, X, X_val, pu, qi, bu, bi, yj):
        """Performs SGD algorithm, learns model weights.
        Args:
            X (ndarray): training set, first column must contains users
                indexes, second one items indexes, and third one ratings.
            X_val (ndarray or `None`): validation set with same structure
                as X.
            pu (ndarray): users latent factor matrix.
            qi (ndarray): items latent factor matrix.
            bu (ndarray): users biases vector.
            bi (ndarray): items biases vector.
            yj (ndarray): The implicit item factors.
        """
        I = [[] for _ in range(self.n_users)]
        for u, i, _ in X:
            I[int(u)].append(int(i))

        # I = [[int(i) for u, i, _ in X if u == user]
        #         for user in np.unique(X[:,0])
        # ]

        self.I = np.full((np.unique(X[:,0]).shape[0], max([len(x) for x in I]) + 1), -1)
        for i, v in enumerate(I):
            self.I[i][0:len(v)] = v

        for epoch_ix in range(self.n_epochs):
            start = self._on_epoch_begin(epoch_ix)

            if self.shuffle:
                X = _shuffle(X)

            pu, qi, bu, bi, yj, train_loss = _run_svdpp_epoch(
                                X, pu, qi, bu, bi, yj, self.global_mean, self.n_factors, self.I.copy(),
                                self.lr_pu, self.lr_qi, self.lr_bu, self.lr_bi, self.lr_yj,
                                self.reg_pu, self.reg_qi, self.reg_bu, self.reg_bi, self.reg_yj
                            )

            if X_val is not None:
                self.metrics_[epoch_ix, :] = _compute_svdpp_val_metrics(X_val, pu, qi, bu, bi, yj,
                                                                  self.global_mean,
                                                                  self.n_factors, self.I)
                self._on_epoch_end(start,
                                   train_loss=train_loss,
                                   val_loss=self.metrics_[epoch_ix, 0],
                                   val_rmse=self.metrics_[epoch_ix, 1],
                                   val_mae=self.metrics_[epoch_ix, 2])

                if self.early_stopping:
                    if self._early_stopping(self.metrics_[:, 1], epoch_ix, self.min_delta_):
                        break

            else:
                self._on_epoch_end(start, train_loss=train_loss)

        self.pu = pu
        self.qi = qi
        self.bu = bu
        self.bi = bi
        self.yj = yj

    @timer(text='\nTraining took ')
    def fit(self, X, X_val=None, i_factor=None, u_factor=None, early_stopping=False, shuffle=False, min_delta=0.001):
        """Learns model weights.
        Args:
            X (pandas DataFrame): training set, must have `u_id` for user id,
                `i_id` for item id and `rating` columns.
            X_val (pandas DataFrame, defaults to `None`): validation set with
                same structure as X.
            i_factor (pandas DataFrame, defaults to `None`): initialization for Qi. The dimension should match self.factor
            u_factor (pandas DataFrame, defaults to `None`): initialization for Pu. The dimension should match self.factor
            early_stopping (boolean): whether or not to stop training based on
                a validation monitoring.
            shuffle (boolean): whether or not to shuffle the training set
                before each epoch.
            min_delta (float, defaults to .001): minimun delta to arg for an
                improvement.
        Returns:
            self (SVD object): the current fitted object.
        """
        self.early_stopping = early_stopping
        self.shuffle = shuffle
        self.min_delta_ = min_delta

        self.users_list = np.unique(X[:, 0])
        self.items_list = np.unique(X[:, 1])

        if X_val is not None:
            self.metrics_ = np.zeros((self.n_epochs, 3), dtype=np.float)

        self.global_mean = np.mean(X[:, 2])

        # Initialize pu, qi, bu, bi, yj
        self.n_users = self.users_list.shape[0]
        self.n_items = self.items_list.shape[0]

        if i_factor is not None:
            qi = i_factor
        else:
            qi = np.random.normal(0, .1, (self.n_items, self.n_factors))

        if u_factor is not None:
            pu = u_factor
        else:
            pu = np.random.normal(0, .1, (self.n_users, self.n_factors))

        bu = np.zeros(self.n_users)
        bi = np.zeros(self.n_items)

        yj = np.random.normal(0, .1, (self.n_items, self.n_factors))

        print('Start training...')
        self._sgd(X, X_val, pu, qi, bu, bi, yj)
        print("Done.")

        return self

    @timer(text='\nTraining took ')
    def load_checkpoint_and_fit(self, checkpoint, X, X_val=None, early_stopping=False, shuffle=False, min_delta=0.001):
        """
        Load a .pkl checkpoint and continue training from that checkpoint
        Args:
            checkpoint (string): path to .pkl checkpoint file.
            X (pandas DataFrame): training set, must have `u_id` for user id,
                `i_id` for item id and `rating` columns.
            X_val (pandas DataFrame, defaults to `None`): validation set with
                same structure as X.
            early_stopping (boolean): whether or not to stop training based on
                a validation monitoring.
            shuffle (boolean): whether or not to shuffle the training set
                before each epoch.
            min_delta (float, defaults to .001): minimun delta to arg for an
                improvement.
        Returns:
            self (SVD object): the current fitted object.
        """
        self.early_stopping = early_stopping
        self.shuffle = shuffle
        self.min_delta_ = min_delta

        # Load parameter from checkpoint
        with open(checkpoint[:-4]+'.pkl', mode='rb') as map_dict:
            data = pickle.load(map_dict)

        pu = data['pu']
        qi = data['qi']
        bu = data['bu']
        bi = data['bi']
        yj = data['yj']

        print(f"Load checkpoint from {checkpoint} successfully.")

        self.users_list = np.unique(self.X[:, 0])
        self.items_list = np.unique(self.X[:, 1])

        if X_val is not None:
            self.metrics_ = np.zeros((self.n_epochs, 3), dtype=np.float)

        self.global_mean = np.mean(X[:, 2])

        print('Start training...')
        self._sgd(X, X_val, pu, qi, bu, bi, yj)
        print("Done.")

        return self

    def save_checkpoint(self, path):
        """Save the model parameter (Pu, Qi, bu, bi)
        and two mapping dictionary (user_dict, item_dict) to a .pkl file.

        Args:
            path (string): path to .npz file.
        """
        checkpoint = {
            'pu' : self.pu,
            'qi' : self.qi,
            'bu' : self.bu,
            'bi' : self.bi,
            'yj' : self.yj
        }

        with open(path, mode='wb') as map_dict:
            pickle.dump(checkpoint, map_dict, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Save checkpoint to {path} successfully.")

    def predict_pair(self, u_id, i_id, clip=True):
        """Returns the model rating prediction for a given user/item pair.
        Args:
            u_id (int): an user id.
            i_id (int): an item id.
            clip (boolean, default is `True`): whether to clip the prediction
                or not.
        Returns:
            pred (float): the estimated rating for the given user/item pair.
        """
        user_known, item_known = False, False
        pred = self.global_mean

        if u_id in self.users_list:
            user_known = True
            pred += self.bu[u_id]

        if i_id in self.items_list:
            item_known = True
            pred += self.bi[i_id]

        if user_known and item_known:
            # Items rated by user u
            last_ele = np.where(self.I[u_id] == -1)[0][0]
            Iu = self.I[u_id, :last_ele]

            # Square root of number of items rated by user u
            sqrt_Iu = np.sqrt(len(Iu))

            # compute user implicit feedback
            u_impl_fdb = np.zeros(self.n_factors)
            for j in Iu:
                for factor in range(self.n_factors):
                    u_impl_fdb[factor] += self.yj[j, factor] / sqrt_Iu

            pred += np.dot(self.qi[i_id], self.pu[u_id] + u_impl_fdb)

        if clip:
            pred = self.max_rating if pred > self.max_rating else pred
            pred = self.min_rating if pred < self.min_rating else pred

        return pred
