import numpy as np
import pandas as pd
from math import sqrt
from scipy import sparse
import time
import pickle

from utils import timer
from .helper import _run_svdpp_epoch, _compute_svdpp_val_metrics, _shuffle
from .svd import svd

class svdpp(svd):
    def __init__(self, learning_rate=.005, lr_pu=None, lr_qi=None, lr_bu=None, lr_bi=None, lr_yj=None,
                 regularization=0.02, reg_pu=None, reg_qi=None, reg_bu=None, reg_bi=None, reg_yj=None,
                 n_epochs=20, n_factors=100, min_rating=1, max_rating=5, i_factor=None, u_factor=None):
        svd.__init__(self, learning_rate, lr_pu, lr_qi, lr_bu, lr_bi,
                 regularization, reg_pu, reg_qi, reg_bu, reg_bi,
                 n_epochs, n_factors, min_rating, max_rating, i_factor, u_factor)
        self.lr_yj = lr_yj if lr_yj is not None else learning_rate
        self.reg_yj = reg_yj if reg_yj is not None else regularization

    def _sgd(self, X, X_val, pu, qi, bu, bi, yj):
        """Performs SGD algorithm, learns model weights.
        Args:
            X (numpy array): training set, first column must contains users
                indexes, second one items indexes, and third one ratings.
            X_val (numpy array or `None`): validation set with same structure
                as X.
            pu (numpy array): users latent factor matrix.
            qi (numpy array): items latent factor matrix.
            bu (numpy array): users biases vector.
            bi (numpy array): items biases vector.
            yj (numpy array): The implicit item factors.
        """
        I = [[int(i) for u, i, _ in X if u == user]
                for user in np.unique(X[:,0])
        ]
        self.I = np.full((np.unique(X[:,0]).shape[0], max([len(x) for x in I])), -1)
        for i, v in enumerate(I):
            self.I[i][0:len(v)] = v

        for epoch_ix in range(self.n_epochs):
            start = self._on_epoch_begin(epoch_ix)

            if self.shuffle:
                X = _shuffle(X)

            pu, qi, bu, bi, yj, train_loss = _run_svdpp_epoch(
                                X, pu, qi, bu, bi, yj, self.global_mean, self.n_factors, self.I,
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

        print('\nPreprocessing data...')
        X = self._preprocess_data(X)
        if X_val is not None:
            self.metrics_ = np.zeros((self.n_epochs, 3), dtype=np.float)
            X_val = self._preprocess_data(X_val, train=False)

        self.global_mean = np.mean(X[:, 2])

        # Initialize pu, qi, bu, bi, yj
        n_user = len(np.unique(X[:, 0]))
        n_item = len(np.unique(X[:, 1]))

        if i_factor is not None:
            qi = i_factor
        else:
            qi = np.random.normal(0, .1, (n_item, self.n_factors))

        if u_factor is not None:
            pu = u_factor
        else:
            pu = np.random.normal(0, .1, (n_user, self.n_factors))

        bu = np.zeros(n_user)
        bi = np.zeros(n_item)

        yj = np.random.normal(0, .1, (n_item, self.n_factors))

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

        self.item_dict = data['item_dict']
        self.user_dict = data['user_dict']
        pu = data['pu']
        qi = data['qi']
        bu = data['bu']
        bi = data['bi']
        yj = data['yj']

        print(f"Load checkpoint from {checkpoint} successfully.")

        print('\nPreprocessing data...')
        X = self._preprocess_data(X, train=False)
        if X_val is not None:
            self.metrics_ = np.zeros((self.n_epochs, 3), dtype=np.float)
            X_val = self._preprocess_data(X_val, train=False)

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
            'user_dict': self.user_dict,
            'item_dict' : self.item_dict,
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

        if u_id in self.user_dict:
            user_known = True
            u_ix = self.user_dict[u_id]
            pred += self.bu[u_ix]

        if i_id in self.item_dict:
            item_known = True
            i_ix = self.item_dict[i_id]
            pred += self.bi[i_ix]

        if user_known and item_known:
            # Items rated by user u
            Iu = self.I[u_ix]
            Iu = Iu[Iu >= 0]
            # Square root of number of items rated by user u
            sqrt_Iu = np.sqrt(len(Iu))

            # compute user implicit feedback
            u_impl_fdb = np.zeros(self.n_factors)
            for j in Iu:
                for factor in range(self.n_factors):
                    u_impl_fdb[factor] += self.yj[j, factor] / sqrt_Iu

            pred += np.dot(self.qi[i_ix], self.pu[u_ix] + u_impl_fdb)

        if clip:
            pred = self.max_rating if pred > self.max_rating else pred
            pred = self.min_rating if pred < self.min_rating else pred

        return pred

    def predict(self, X):
        """Returns estimated ratings of several given user/item pairs.
        Args:
            X (pandas DataFrame): storing all user/item pairs we want to
                predict the ratings. Must contains columns labeled `u_id` and
                `i_id`.
        Returns:
            predictions: list, storing all predictions of the given user/item
                pairs.
        """
        predictions = []

        for u_id, i_id in zip(X['u_id'], X['i_id']):
            predictions.append(self.predict_pair(u_id, i_id))

        return predictions