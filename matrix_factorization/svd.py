import numpy as np
import pandas as pd
from math import sqrt
from scipy import sparse
import time
import pickle

from utils import timer
from .helper import _run_svd_epoch, _compute_svd_val_metrics, _shuffle, _calculate_precision_recall


class SVD:
    """Implements Simon Funk SVD algorithm engineered during the Netflix Prize.

    Attributes:
        lr (float): learning rate.
        reg (float): regularization factor.
        n_epochs (int): number of SGD iterations.
        n_factors (int): number of latent factors.
        global_mean (float): ratings arithmetic mean.
        pu (ndarray): users latent factor matrix.
        qi (ndarray): items latent factor matrix.
        bu (ndarray): users biases vector.
        bi (ndarray): items biases vector.
        early_stopping (boolean): whether or not to stop training based on a validation monitoring.
        shuffle (boolean): whether or not to shuffle data before each epoch.
    """

    def __init__(self, learning_rate=.005, lr_pu=None, lr_qi=None, lr_bu=None, lr_bi=None,
                 regularization=0.02, reg_pu=None, reg_qi=None, reg_bu=None, reg_bi=None,
                 n_epochs=20, n_factors=100, min_rating=0.5, max_rating=5):
        """SVD model

        Args:
            learning_rate (float, optional): the common learning rate. Defaults to .005.
            lr_pu (float, optional): Pu's specific learning rate. Defaults to learning_rate.
            lr_qi (float, optional): Qi's specific learning rate. Defaults to learning_rate.
            lr_bu (float, optional): bu's specific learning rate. Defaults to learning_rate.
            lr_bi (float, optional): bi's specific learning rate. Defaults to learning_rate.
            regularization (float, optional): the common regularization term. Defaults to 0.02.
            reg_pu (float, optional): Pu's specific regularization term. Defaults to regularization.
            reg_qi (float, optional): Qi's specific regularization term. Defaults to regularization.
            reg_bu (float, optional): bu's specific regularization term. Defaults to regularization.
            reg_bi (float, optional): bi's specific regularization term. Defaults to regularization.
            n_epochs (int, optional): number of SGD iterations. Defaults to 20.
            n_factors (int, optional): number of latent factors. Defaults to 100.
            min_rating (float, optional): the minimum value of predicted rating. Defaults to 0.5.
            max_rating (float, optional): the maximum value of predicted rating. Defaults to 5.
        """

        self.lr_pu = lr_pu if lr_pu is not None else learning_rate
        self.lr_qi = lr_qi if lr_qi is not None else learning_rate
        self.lr_bu = lr_bu if lr_bu is not None else learning_rate
        self.lr_bi = lr_bi if lr_bi is not None else learning_rate

        self.reg_pu = reg_pu if reg_pu is not None else regularization
        self.reg_qi = reg_qi if reg_qi is not None else regularization
        self.reg_bu = reg_bu if reg_bu is not None else regularization
        self.reg_bi = reg_bi if reg_bi is not None else regularization

        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.early_stopping = False
        self.shuffle = False
        self.global_mean = np.nan
        self.metrics_ = None
        self.min_delta_ = 0.001

    def _sgd(self, X, X_val, pu, qi, bu, bi):
        """Performs SGD algorithm, learns model weights.

        Args:
            X (ndarray): training set, first column must contains users
                indexes, second one items indexes, and third one ratings.
            X_val (ndarray, optional): validation set with same structure as X. Defaults to None.
            pu (ndarray): users latent factor matrix.
            qi (ndarray): items latent factor matrix.
            bu (ndarray): users biases vector.
            bi (ndarray): items biases vector.
        """
        for epoch_ix in range(self.n_epochs):
            start = self._on_epoch_begin(epoch_ix)

            if self.shuffle:
                X = _shuffle(X)

            pu, qi, bu, bi, train_loss = _run_svd_epoch(
                                X, pu, qi, bu, bi, self.global_mean, self.n_factors,
                                self.lr_pu, self.lr_qi, self.lr_bu, self.lr_bi,
                                self.reg_pu, self.reg_qi, self.reg_bu, self.reg_bi
                            )

            if X_val is not None:
                self.metrics_[epoch_ix, :] = _compute_svd_val_metrics(X_val, pu, qi, bu, bi,
                                                                  self.global_mean,
                                                                  self.n_factors)
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

    @timer(text='\nTraining took ')
    def fit(self, X, X_val=None, i_factor=None, u_factor=None, early_stopping=False, shuffle=False, min_delta=0.001):
        """Learns model weights.

        Args:
            X (ndarray): training set, must have `u_id` for user id,
                `i_id` for item id and `rating` columns.
            X_val (ndarray): validation set with same structure as X. Defaults to `None`
            i_factor (ndarray): initialization for Qi. Defaults to `None`
            u_factor (ndarray): initialization for Pu. Defaults to `None`
            early_stopping (boolean): whether or not to stop training based on a validation monitoring.
            shuffle (boolean): whether or not to shuffle the training set before each epoch.
            min_delta (float, defaults to .001): minimun delta to arg for an improvement.

        Returns:
            self (SVD object): the current fitted object.
        """
        X = X.copy()

        self.early_stopping = early_stopping
        self.shuffle = shuffle
        self.min_delta_ = min_delta

        if X_val is not None:
            X_val = X_val.copy()
            self.metrics_ = np.zeros((self.n_epochs, 3), dtype=np.float)

        self.global_mean = np.mean(X[:, 2])

        self.users_list = np.unique(X[:, 0])
        self.items_list = np.unique(X[:, 1])
        n_user = self.users_list.shape[0]
        n_item = self.items_list.shape[0]

        # Initialize pu, qi, bu, bi
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

        print('Start training...')
        self._sgd(X, X_val, pu, qi, bu, bi)
        print("Done.")

        return self

    @timer(text='\nTraining took ')
    def load_checkpoint_and_fit(self, checkpoint, X, X_val=None, early_stopping=False, shuffle=False, min_delta=0.001):
        """
        Load a checkpoint and continue training from that checkpoint.
        The model should be save as two separate file with the same path and the same name, one .npz and one .pkl

        Args:
            checkpoint (string): path to .pkl checkpoint file.
            X (ndarray): training set, must have first column for user id, second column for item id and the last is rating column.
            X_val (ndarray): validation set with same structure as X. Defaults to None.
            early_stopping (boolean): whether or not to stop training based on a validation monitoring.
            shuffle (boolean): whether or not to shuffle the training set before each epoch.
            min_delta (float, defaults to .001): minimun delta to arg for an improvement.

        Returns:
            self (SVD object): the current fitted object.
        """
        self.early_stopping = early_stopping
        self.shuffle = shuffle
        self.min_delta_ = min_delta

        # Load parameter from checkpoint
        with open(checkpoint[:-4]+'.pkl', mode='rb') as map_dict:
            data = pickle.load(map_dict)

        self.users_list = np.unique(X[:, 0])
        self.items_list = np.unique(X[:, 1])

        pu = data['pu']
        qi = data['qi']
        bu = data['bu']
        bi = data['bi']

        print(f"Load checkpoint from {checkpoint} successfully.")

        if X_val is not None:
            self.metrics_ = np.zeros((self.n_epochs, 3), dtype=np.float)

        self.global_mean = np.mean(X[:, 2])

        print('Start training...')
        self._sgd(X, X_val, pu, qi, bu, bi)
        print("Done.")

        return self

    def save_checkpoint(self, path):
        """Save the model parameter (Pu, Qi, bu, bi) to a .pkl file.

        Args:
            path (string): path to .npz file.
        """
        checkpoint = {
            'pu' : self.pu,
            'qi' : self.qi,
            'bu' : self.bu,
            'bi' : self.bi
        }

        with open(path, mode='wb') as map_dict:
            pickle.dump(checkpoint, map_dict, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Save checkpoint to {path} successfully.")

    def predict_pair(self, u_id, i_id):
        """Returns the model rating prediction for a given user/item pair.

        Args:
            u_id (int): a user id.
            i_id (int): an item id.

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
            pred += np.dot(self.pu[u_id], self.qi[i_id])

        return pred

    def predict(self, X, clip=True):
        """Returns estimated ratings of several given user/item pairs.

        Args:
            X (ndarray): storing all user/item pairs we want to predict the ratings. Must have first column for user id, second column for item id.
            clip (boolean, default is `True`): whether to clip the prediction or not.

        Returns:
            predictions (ndarray): Storing all predictions of the given user/item pairs. The first column is user id, the second column is item id, the third column is the observed rating, and the forth column is the predicted rating.
        """
        X = X.copy()
        n_pairs = X.shape[0]

        self.predictions = np.zeros((n_pairs, X.shape[1]+1))
        self.predictions[:, :3] = X

        for pair in range(n_pairs):
            self.predictions[pair, 3] = self.predict_pair(X[pair, 0].astype(int), X[pair, 1].astype(int))

        if clip:
            np.clip(self.predictions[:, 3], self.min_rating, self.max_rating, out=self.predictions[:, 3])

        return self.predictions[:, 3]

    def rmse(self):
        """Calculate Root Mean Squared Error between the predictions and the ground truth.
        Print the RMSE.
        """
        mse = np.mean((self.predictions[:, 2] - self.predictions[:, 3])**2)
        rmse_ = np.sqrt(mse)
        print(f"RMSE: {rmse_:.5f}")

    def mae(self):
        """Calculate Mean Absolute Error between the predictions and the ground truth.
        Print the MAE.
        """
        mae_ = np.mean(np.abs(self.predictions[:, 2] - self.predictions[:, 3]))
        print(f"MAE: {mae_:.5f}")

    def precision_recall_at_k(self, k=10, threshold=3.5):
        """Calculate the precision and recall at k metrics.
        Args:
            k (int, optional): the k metric. Defaults to 10.
            threshold (float, optional): relevent threshold. Defaults to 3.5.
        """

        n_users = self.users_list.shape[0]

        # First map the predictions to each user.
        user_est_true = [ [] for _ in range(n_users)]
        for u_id, _, true_r, est in self.predictions:
            user_est_true[int(u_id)].append([est, true_r])

        # precision and recall at k metrics for each user
        precisions = np.zeros(n_users)
        recalls = np.zeros(n_users)

        for u_id, user_ratings in enumerate(user_est_true):
            precisions[u_id], recalls[u_id] = _calculate_precision_recall(np.array(user_ratings), k, threshold)

        precision = sum(prec for prec in precisions) / n_users
        recall = sum(rec for rec in recalls) / n_users

        print(f"Precision: {precision:.5f}")
        print(f"Recall: {recall:.5f}")

    def _early_stopping(self, list_val_rmse, epoch_idx, min_delta):
        """Returns True if validation rmse is not improving.
        Last rmse (plus `min_delta`) is compared with the second to last.

        Agrs:
            list_val_rmse (list): ordered validation RMSEs.
            min_delta (float): minimun delta to arg for an improvement.

        Returns:
            (boolean): whether or not to stop training.
        """
        if epoch_idx > 0:
            if list_val_rmse[epoch_idx] + min_delta > list_val_rmse[epoch_idx-1]:
                self.metrics_ = self.metrics_[:(epoch_idx+1), :]
                return True
        return False

    def _on_epoch_begin(self, epoch_ix):
        """Displays epoch starting log and returns its starting time.

        Args:
            epoch_ix: integer, epoch index.

        Returns:
            start (float): starting time of the current epoch.
        """
        start = time.time()
        end = '  | ' if epoch_ix < 9 else ' | '
        print('Epoch {}/{}'.format(epoch_ix + 1, self.n_epochs), end=end)

        return start

    def _on_epoch_end(self, start, train_loss, val_loss=None, val_rmse=None, val_mae=None):
        """
        Displays epoch ending log. If self.verbose compute and display
        validation metrics (loss/rmse/mae).

        Args:
            start (float): starting time of the current epoch.
            train_loss: float, training loss
            val_loss: float, validation loss
            val_rmse: float, validation rmse
            val_mae: float, validation mae
        """
        end = time.time()

        print('train_loss: {:.5f}'.format(train_loss), end=' - ')
        if val_loss is not None:
            print('val_loss: {:.5f}'.format(val_loss), end=' - ')
            print('val_rmse: {:.5f}'.format(val_rmse), end=' - ')
            print('val_mae: {:.5f}'.format(val_mae), end=' - ')

        print('took {:.2f} sec'.format(end - start))

    def get_val_metrics(self):
        """Get validation metrics

        Returns:
            a Pandas DataFrame
        """
        if isinstance(self.metrics_, np.ndarray) and (self.metrics_.shape[1] == 3):
            return pd.DataFrame(self.metrics_, columns=["Loss", "RMSE", "MAE"])
