import numpy as np

from utils import timer
from .kNN import kNN
from .knn_helper import _baseline_sgd, _baseline_als, _predict_baseline


class kNNBaseline(kNN):
    """Reimplementation of kNNBaseline alrgorithm.

    Args:
        k (int): Number of neibors use in prediction
        min_k (int): The minimum number of neighbors to take into account for aggregation. If there are not enough neighbors, the neighbor aggregation is set to zero
        distance (str, optional): Distance function. Defaults to `"cosine"`.
        uuCF (boolean, optional): True if using user-based CF, False if using item-based CF. Defaults to `False`.
        verbose (boolean): Show predicting progress. Defaults to `False`.
        awareness_constrain (boolean): If `True`, the model must aware of all users and items in the test set, which means that these users and items are in the train set as well. This constrain helps speed up the predicting process (up to 1.5 times) but if a user of an item is unknown, kNN will fail to give prediction. Defaults to `False`.
    """

    def fit(self, train_data, similarity_measure="cosine", genome=None, similarity_matrix=None, baseline_options={'method':'als','n_epochs': 10,'reg_u':15,'reg_i':10}):
        """Fit data (utility matrix) into the predicting model.

        Args:
            data (ndarray): Training data.
            similarity_measure (str, optional): Similarity measure function. Defaults to "cosine".
            genome (ndarray): Movie genome scores from MovieLens 20M. Defaults to "None".
            similarity_matrix (ndarray): Pre-calculate similarity matrix.  Defaults to "None".
            baseline_options(dict): Used to configure how to compute baseline estimate.
        """
        kNN.fit(self, train_data, similarity_measure, genome, similarity_matrix)
        self.__baseline(baseline_options)

    def predict_pair(self, x_id, y_id):
        """Predict the rating of user u for item i

        Args:
            x_id (int): index of x (For uuCF, x -> user)
            y_id (int): index of y (For uuCF, y -> item)

        Returns:
            pred (float): prediction of the given user/item pair.
        """
        if not self.awareness_constrain:
            pred = self.global_mean

            x_known, y_known = False, False
            if x_id in self.x_list:
                x_known = True
                pred += self.bx[x_id]
            if y_id in self.y_list:
                y_known = True
                pred += self.by[y_id]

            if not (x_known and y_known):
                return pred

        pred = _predict_baseline(x_id, y_id, self.x_rated[y_id], self.S, self.k, self.min_k, self.global_mean, self.bx, self.by)
        return pred

    @timer("Time for computing the baseline estimate: ")
    def __baseline(self, baseline_options):
        """Compute the baseline estimate for all user and movie using the following fomular.
        b_{ui} = \mu + b_u + b_i
        """
        if baseline_options['method'] == 'als':
            self.bx, self.by = _baseline_als(
                self.global_mean
                , self.n_x, self.n_y
                , self.x_rated, self.y_ratedby
                , baseline_options['n_epochs']
                , baseline_options['reg_u']
                , baseline_options['reg_i']
            )
        elif baseline_options['method'] == 'sgd':
            self.bx, self.by = _baseline_sgd(
                self.X, self.global_mean
                , self.n_x, self.n_y
                , baseline_options['n_epochs']
                , baseline_options['learning_rate']
                , baseline_options['regularization']
            )
