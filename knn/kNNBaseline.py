import numpy as np

from utils import timer
from .kNN import kNN
from .knn_helper import _predict_baseline
from .similarities import _pcc_baseline
from .baseline_helper import _run_baseline_sgd, _run_baseline_als


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

    def fit(self, train_set, similarity_measure="cosine", genome=None, shrinkage=100, similarity_matrix=None, baseline_options={'method':'als','n_epochs': 10,'reg_u':15,'reg_i':10}):
        """Fit data (utility matrix) into the predicting model.

        Args:
            data (ndarray): Training data.
            similarity_measure (str, optional): Similarity measure function. Defaults to "cosine".
            genome (ndarray): Movie genome scores from MovieLens 20M. Defaults to "None".
            shrinkage (int): only used in computing the (shrunk) Pearson correlation coefficient. Defaults to 100.
            similarity_matrix (ndarray): Pre-calculate similarity matrix.  Defaults to "None".
            baseline_options(dict): Used to configure how to compute baseline estimate.
        """
        self.fit_train_set(train_set)

        self.list_ur_ir()

        self.baseline(baseline_options)

        self.supported_sim_func = ["cosine", "pearson", "pearson_baseline"]
        self.compute_similarity_matrix(similarity_measure, genome, shrinkage, similarity_matrix)

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
    def baseline(self, baseline_options):
        """Compute the baseline estimate for all user and movie using the following fomular.
        b_{ui} = \mu + b_u + b_i
        """
        bx = np.zeros(self.n_x)
        by = np.zeros(self.n_y)

        self.__supported_baseline_optimizer = ['als', 'sgd']
        assert baseline_options['method'] in self.__supported_baseline_optimizer, f"Similarity measure function should be one of {self.__supported_baseline_optimizer}"

        if baseline_options['method'] == 'als':
            n_epochs = baseline_options.get('n_epochs', 10)
            reg_x = baseline_options.get('reg_u', 15)
            reg_y = baseline_options.get('reg_i', 10)
            if not self.uuCF:
                reg_x, reg_y = reg_y, reg_x

            for epoch in range(n_epochs):
                bx, by = _run_baseline_als(
                    bx, by, self.global_mean
                    , self.n_x, self.n_y
                    , self.x_rated, self.y_ratedby
                    , reg_x, reg_y
                )

        elif baseline_options['method'] == 'sgd':
            n_epochs = baseline_options.get('n_epochs', 20)
            lr = baseline_options.get('learning_rate', 0.005)
            reg = baseline_options.get('regularization', 0.02)

            for epoch in range(n_epochs):
                bx, by = _run_baseline_sgd(
                    self.X, bx, by, self.global_mean
                    , lr, reg
                )

        self.bx, self.by = bx, by

    def compute_similarity_matrix(self, similarity_measure, genome, shrinkage, similarity_matrix):
        """Computing the similarity matrix for kNN.

        Args:
            similarity_measure (str, optional): Similarity measure function. Defaults to "cosine".
            genome (ndarray, optional): Movie genome scores from MovieLens 20M. If "None", the similarity matrix will be computed using rating information, else using the genome vectors to calculate. Defaults to "None".
            shrinkage (int): only used in computing the (shrunk) Pearson correlation coefficient. Defaults to 100.
            similarity_matrix (ndarray, optional): Pre-calculate similarity matrix.  Defaults to "None".
        """

        kNN.compute_similarity_matrix(self, similarity_measure, genome, similarity_matrix)

        if self.S is not None:
            return

        if similarity_measure == "pearson_baseline":
            self.S = _pcc_baseline(self.n_x, self.x_rated, self.global_mean, self.bx, self.by, shrinkage, min_support=self.min_k)
