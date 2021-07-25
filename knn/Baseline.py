import numpy as np

from utils import timer
from .baseline_helper import _run_baseline_sgd, _run_baseline_als, _compute_baseline_val_metrics


class Baseline:
    """Baseline algorithm, which optimizes the user- and item- specific biases.
    """

    def __init__(self):
        self.x_rated = None
        self.y_ratedby = None

    def fit(self, train_set, val_set=None, baseline_options={'method':'als','n_epochs': 10,'reg_u':15,'reg_i':10}):
        """Fit data (utility matrix) into the predicting model.

        Args:
            train_set (ndarray): Training data.
            val_set (ndarray): Validating data.
            baseline_options(dict): Used to configure how to compute baseline estimate.
        """
        self.X = train_set.copy()
        self.X_val = val_set.copy()

        self.n_x = len(np.unique(self.X[:, 0]))       # For uuCF, x -> user
        self.n_y = len(np.unique(self.X[:, 1]))       # For uuCF, y -> item

        self.global_mean = np.mean(self.X[:, 2])
        self.baseline(baseline_options)

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

            if self.x_rated is None and self.y_ratedby is None:
                print("Listing all users rated each item, and all items rated by each user ...")
                self.list_ur_ir()

            for epoch in range(n_epochs):
                bx, by = _run_baseline_als(
                    bx, by, self.global_mean
                    , self.n_x, self.n_y
                    , self.x_rated, self.y_ratedby
                    , reg_x, reg_y
                )
                self.validate(epoch, bx, by)

        elif baseline_options['method'] == 'sgd':
            n_epochs = baseline_options.get('n_epochs', 20)
            lr = baseline_options.get('learning_rate', 0.005)
            reg = baseline_options.get('regularization', 0.02)

            for epoch in range(n_epochs):
                bx, by = _run_baseline_sgd(
                    self.X, bx, by, self.global_mean
                    , lr, reg
                )
                self.validate(epoch, bx, by)

        self.bx, self.by = bx, by

    def validate(self, epoch, bx, by):
        if self.X_val is not None:
            val_loss, val_rmse, val_mae = _compute_baseline_val_metrics(self.X_val, self.global_mean, bx, by)
            print(f"Epoch {epoch}", end=': ')
            print('val_loss: {:.5f}'.format(val_loss), end=' - ')
            print('val_rmse: {:.5f}'.format(val_rmse), end=' - ')
            print('val_mae: {:.5f}'.format(val_mae))

    @timer("Listing took ")
    def list_ur_ir(self):
        """Listing all users rated each item, and all items rated by each user.
        If uuCF, x denotes user and y denotes item.
        All users who rated each item are stored in list `x_rated`.
        All items who rated by each user are stored in list `y_ratedby`.
        The denotation are reversed if iiCF.
        """
        self.x_rated = [[] for _ in range(self.n_y)]        # List where element `i` is ndarray of `(x, rating)` where `x` is all x that rated y, and the ratings.
        self.y_ratedby = [[] for _ in range(self.n_x)]      # List where element `i` is ndarray of `(y, rating)` where `y` is all y that rated by x, and the ratings.

        for xid, yid, r in self.X:
            self.x_rated[int(yid)].append([xid, r])
            self.y_ratedby[int(xid)].append([yid, r])

        for yid in range(self.n_y):
            self.x_rated[yid] = np.array(self.x_rated[yid])
        for xid in range(self.n_x):
            self.y_ratedby[xid] = np.array(self.y_ratedby[xid])
