import numpy as np

from scipy import sparse

from utils import timer
from .kNN import kNN
from .knn_helper import _predict_mean


class kNNwithMean(kNN):
    """Reimplementation of kNN with mean alrgorithm.

    Args:
        k (int): Number of neibors use in prediction
        min_k (int): The minimum number of neighbors to take into account for aggregation. If there are not enough neighbors, the neighbor aggregation is set to zero
        uuCF (boolean, optional): True if using user-based CF, False if using item-based CF. Defaults to `False`.
        verbose (boolean): Show predicting progress. Defaults to `False`.
        awareness_constrain (boolean): If `True`, the model must aware of all users and items in the test set, which means that these users and items are in the train set as well. This constrain helps speed up the predicting process (up to 1.5 times) but if a user of an item is unknown, kNN will fail to give prediction. Defaults to `False`.
    """

    def fit(self, train_set, similarity_measure="cosine", genome=None, similarity_matrix=None):
        """Fit data (utility matrix) into the predicting model.

        Args:
            data (ndarray): Training data.
            similarity_measure (str, optional): Similarity measure function. Defaults to "cosine".
            genome (ndarray): Movie genome scores from MovieLens 20M. Defaults to "None".
            similarity_matrix (ndarray): Pre-calculate similarity matrix.  Defaults to "None".
        """
        kNN.fit(self, train_set, similarity_measure, genome, similarity_matrix)

        self.utility = sparse.csr_matrix((
                self.X[:, 2],
                (self.X[:, 0].astype(int), self.X[:, 1].astype(int))
            ))

        self.__mean_normalize()

    def predict_pair(self, x_id, y_id):
        """Predict the rating of user u for item i

        Args:
            x_id (int): index of x (For uuCF, x -> user)
            y_id (int): index of y (For uuCF, y -> item)

        Returns:
            pred (float): prediction of the given user/item pair.
        """
        if not self.awareness_constrain:
            x_known, y_known = False, False

            if x_id in self.x_list:
                x_known = True
            if y_id in self.y_list:
                y_known = True

            if not (x_known and y_known):
                # if self.uuCF:
                #     print(f"Can not predict rating of user {x_id} for item {y_id}.")
                # else:
                #     print(f"Can not predict rating of user {y_id} for item {x_id}.")
                return self.global_mean

        pred = _predict_mean(x_id, y_id, self.x_rated[y_id], self.mu, self.S, self.k, self.min_k)
        return pred + self.mu[x_id]

    def __mean_normalize(self):
        """Normalize the utility matrix.
        This method only normalize the data base on the mean of ratings.
        Any unrated item will remain the same.
        """
        tot = np.array(self.utility.sum(axis=1).squeeze())[0]
        cts = np.diff(self.utility.indptr)
        cts[cts == 0] = 1       # Avoid dividing by 0 resulting nan.

        # Mean ratings of each users.
        self.mu = tot / cts

        # Diagonal matrix with the means on the diagonal.
        d = sparse.diags(self.mu, 0)

        # A matrix that is like Utility, but has 1 at the non-zero position instead of the ratings.
        b = self.utility.copy()
        b.data = np.ones_like(b.data)

        # d*b = Mean matrix - a matrix with the means of each row at the non-zero position
        # Subtract the mean matrix to get the normalize data.
        self.utility -= d*b
