import progressbar

import numpy as np

from utils import timer
from .similarities import _cosine, _pcc, _cosine_genome, _pcc_genome
from .knn_helper import _predict, _calculate_precision_recall, _calculate_ndcg


class kNN:
    """Reimplementation of basic kNN alrgorithm.

    Args:
        min_k (int): The minimum number of neighbors to take into account for aggregation. If there are not enough neighbors, the neighbor aggregation is set to zero
        uuCF (boolean, optional): True if using user-based CF, False if using item-based CF. Defaults to `False`.
        verbose (boolean): Show predicting progress. Defaults to `False`.
        awareness_constrain (boolean): If `True`, the model must aware of all users and items in the test set, which means that these users and items are in the train set as well. This constrain helps speed up the predicting process (up to 1.5 times) but if a user of an item is unknown, kNN will fail to give prediction. Defaults to `False`.
        S (ndarray): Similarity matrix. Initialized to None.
    """
    def __init__(self, min_k=1, uuCF=False, verbose=False, awareness_constrain=False):
        self.min_k = min_k

        self.uuCF = uuCF

        self.verbose = verbose
        self.awareness_constrain = awareness_constrain

        self.S = None

    def fit(self, train_set, similarity_measure="cosine", genome=None, similarity_matrix=None):
        """Fit data (utility matrix) into the predicting model.

        Args:
            train_set (ndarray): Training data.
            similarity_measure (str, optional): Similarity measure function. Defaults to "cosine".
            genome (ndarray, optional): Movie genome scores from MovieLens 20M. Defaults to "None".
            similarity_matrix (ndarray, optional): Pre-calculate similarity matrix.  Defaults to "None".
        """
        self.fit_train_set(train_set)

        self.list_ur_ir()

        self.supported_sim_func = ["cosine", "pearson"]
        self.compute_similarity_matrix(similarity_measure, genome, similarity_matrix)

    @timer("Time for predicting: ")
    def predict(self, test_set, k, min_rating=0.5, max_rating=5, clip=True):
        """Returns estimated ratings of several given user/item pairs.
        Args:
            test_set (adarray): storing all user/item pairs we want to predict the ratings.
            k (int): Number of neighbors use in prediction
            min_rating (float): the minimum value for rating prediction.
            max_rating (float): the maximum value for rating prediction.
            clip (boolean): if True, clip the prediction based on the min and max value.
        Returns:
            predictions (ndarray): Storing all predictions of the given user/item pairs. The first column is user id, the second column is item id, the third column is the observed rating, and the forth column is the predicted rating.
        """
        self.k = k

        test_set = test_set.copy()
        if not self.uuCF:
            test_set[:, [0, 1]] = test_set[:, [1, 0]]     # Swap user_id column to movie_id column if using iiCF

        n_pairs = test_set.shape[0]

        self.predictions = np.zeros((n_pairs, test_set.shape[1]+1))
        self.predictions[:, :3] = test_set

        print(f"Predicting {n_pairs} pairs of user-item with k={self.k} ...")

        if self.verbose:
            bar = progressbar.ProgressBar(maxval=n_pairs, widgets=[progressbar.Bar(), ' ', progressbar.Percentage()])
            bar.start()
            for pair in range(n_pairs):
                self.predictions[pair, 3] = self.predict_pair(test_set[pair, 0].astype(int), test_set[pair, 1].astype(int))
                bar.update(pair + 1)
            bar.finish()
        else:
            for pair in range(n_pairs):
                self.predictions[pair, 3] = self.predict_pair(test_set[pair, 0].astype(int), test_set[pair, 1].astype(int))

        if clip:
            np.clip(self.predictions[:, 3], min_rating, max_rating, out=self.predictions[:, 3])

        return self.predictions

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

        return _predict(x_id, y_id, self.x_rated[y_id], self.S, self.k, self.min_k)

    def __recommend(self, u):
        """Determine all items should be recommended for user u. (uuCF = 1)
        or all users who might have interest on item u (uuCF = 0)
        The decision is made based on all i such that: self.pred(u, i) > 0.
        Suppose we are considering items which have not been rated by u yet.
        NOT YET IMPLEMENTED...

        Args:
            u (int): user that we are recommending

        Returns:
            list: a list of movie that might suit user u
        """
        pass

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
            k (int, optional): number of ranks to use when calculating NDCG.. Defaults to 10.
            threshold (float, optional): relevent threshold. Defaults to 3.5.
        """

        if self.uuCF:
            n_users = self.n_x
        else:
            n_users = self.n_y

        # First map the predictions to each user.
        user_est_true = [ [] for _ in range(n_users) ]
        for x_id, y_id, true_r, est in self.predictions:
            if self.uuCF:
                u_id = int(x_id)
            else:
                u_id = int(y_id)
            user_est_true[u_id].append([est, true_r])

        # precision and recall at k metrics for each user
        precisions = np.zeros(n_users)
        recalls = np.zeros(n_users)

        for u_id, user_ratings in enumerate(user_est_true):
            precisions[u_id], recalls[u_id] = _calculate_precision_recall(np.array(user_ratings), k, threshold)

        precision = sum(prec for prec in precisions) / n_users
        recall = sum(rec for rec in recalls) / n_users

        print(f"Precision@{k}: {precision:.5f} - Recall@{k}: {recall:.5f}")

    def ndcg_at_k(self, k=10):
        """Calculate normalized discounted cumulative gain on the test set.

        Args:
            k (int, optional): number of ranks to use when calculating NDCG. Defaults to 10.
        """

        if self.uuCF:
            n_users = self.n_x
        else:
            n_users = self.n_y

        # First map the predictions to each user.
        users_est_rating = [ [] for _ in range(n_users) ]
        users_true_rating = [ [] for _ in range(n_users) ]
        for x_id, y_id, true_r, est in self.predictions:
            if self.uuCF:
                u_id = int(x_id)
            else:
                u_id = int(y_id)
            users_est_rating[u_id].append(est)
            users_true_rating[u_id].append(true_r)

        # ndcg score of each user
        ndcgs = np.zeros(n_users)

        for u_id in range(n_users):
            ndcgs[u_id] = _calculate_ndcg(np.array(users_true_rating[u_id]), np.array(users_est_rating[u_id]), k)

        ndcg = sum(ndcg_score for ndcg_score in ndcgs) / n_users

        print(f"NDCG@{k}: {ndcg:.5f}")

    def get_top_N_recommendation(self, n=10, criterion='estimated_rating'):
        """Return the top-N recommendation for each user from a set of predictions.

        Args:
            n (int, optional): The number of recommendation to output for each user. Defaults to 10.
            criterion (str, optional): The criterion to get top-N recommendation. Only \"observed_rating\" or \"estimated_rating\" are accepted. Defaults to 'estimated_rating'.

        Returns:
            (list): top-N recommendation for each user.
        """
        assert criterion in ['observed_rating', 'estimated_rating'], "The critiation should be \"observed_rating\" or \"estimated_rating\"."

        if self.uuCF:
            n_users = self.n_x
        else:
            n_users = self.n_y

        if criterion == 'estimated_rating':
            criterion_idx = 3
        elif criterion == 'observed_rating':
            criterion_idx = 3

        top_N = [ [] for _ in range(n_users) ]
        for row in self.predictions:
            if self.uuCF:
                u_id, i_id = int(row[0]), int(row[1])
            else:
                u_id, i_id = int(row[1]), int(row[0])

            top_N[u_id].append([i_id, row[criterion_idx]])

        for u_id, ratings in enumerate(top_N):
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_N[u_id] = ratings[:n]

        return top_N

    def fit_train_set(self, train_set):
        """Get useful attribute from the training set such as rating mean, user and item list, number of users and items.

        Args:
            train_set (ndarray): Training data.
        """
        self.X = train_set.copy()

        if not self.uuCF:
            self.X[:, [0, 1]] = self.X[:, [1, 0]]     # Swap user_id column to movie_id column if using iiCF

        self.global_mean = np.mean(self.X[:, 2])

        self.x_list = np.unique(self.X[:, 0])       # For uuCF, x -> user
        self.y_list = np.unique(self.X[:, 1])       # For uuCF, y -> item

        self.n_x = len(self.x_list)
        self.n_y = len(self.y_list)

    @timer("Listing took ")
    def list_ur_ir(self):
        """Listing all users rated each item, and all items rated by each user.
        If uuCF, x denotes user and y denotes item.
        All users who rated each item are stored in list `x_rated`.
        All items who rated by each user are stored in list `y_ratedby`.
        The denotation are reversed if iiCF.
        """

        print("Listing all users rated each item, and all items rated by each user ...")

        self.x_rated = [[] for _ in range(self.n_y)]        # List where element `i` is ndarray of `(x, rating)` where `x` is all x that rated y, and the ratings.
        self.y_ratedby = [[] for _ in range(self.n_x)]      # List where element `i` is ndarray of `(y, rating)` where `y` is all y that rated by x, and the ratings.

        for xid, yid, r in self.X:
            self.x_rated[int(yid)].append([xid, r])
            self.y_ratedby[int(xid)].append([yid, r])

        for yid in range(self.n_y):
            self.x_rated[yid] = np.array(self.x_rated[yid])
        for xid in range(self.n_x):
            self.y_ratedby[xid] = np.array(self.y_ratedby[xid])

    def compute_similarity_matrix(self, similarity_measure, genome, similarity_matrix):
        """Computing the similarity matrix for kNN.

        Args:
            similarity_measure (str, optional): Similarity measure function. Defaults to "cosine".
            genome (ndarray, optional): Movie genome scores from MovieLens 20M. If "None", the similarity matrix will be computed using rating information, else using the genome vectors to calculate. Defaults to "None".
            similarity_matrix (ndarray, optional): Pre-calculate similarity matrix.  Defaults to "None".
        """

        if similarity_matrix is not None:
            self.S = similarity_matrix      # Directly assign the precomputed similarity matrix to self.S
            return

        print('Computing similarity matrix ...')
        assert similarity_measure in self.supported_sim_func, f"Similarity measure function should be one of {self.supported_sim_func}"
        self.similarity_measure = similarity_measure

        if self.similarity_measure == "cosine":
            if genome is not None:
                self.S = _cosine_genome(genome)
            else:
                self.S = _cosine(self.n_x, self.x_rated, min_support=self.min_k)

        if self.similarity_measure == "pearson":
            if genome is not None:
                self.S = _pcc_genome(genome)
            else:
                self.S = _pcc(self.n_x, self.x_rated, min_support=self.min_k)
