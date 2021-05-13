import pandas as pd
import numpy as np
from scipy import sparse


class DataLoader:
    """
    Contain methods to read and split dataset into training set and test set.
    All dataset is preprocessed by mapping ID according to the training set and return as `ndarray`.
    (IDs which are not included in the training set remain the same).

    Args:
        data_folder (string): Path to folder that contain dataset
        genome_folder (string): Path to folder that contain the genome_scores file
    """
    def __init__(self, data_folder, genome_folder=None):
        self.__data_folder = data_folder
        if genome_folder is None:
            self.__genome_folder = data_folder
        else:
            self.__genome_folder = genome_folder

        self.__train_data = None
        self.__val_data = None
        self.__test_data = None

    def __read_trainset(self, columns):
        self.__train_data = pd.read_csv(
            self.__data_folder + "/rating_train.csv",
            header=0, names=columns
        )

        self.user_dict = {uIds: idx for idx, uIds in enumerate(np.sort(self.__train_data['u_id'].unique()))}
        self.item_dict = {iIds: idx for idx, iIds in enumerate(np.sort(self.__train_data['i_id'].unique()))}
        self.__train_data = self.__preprocess(self.__train_data)

    def __read_testset(self, columns):
        self.__test_data = pd.read_csv(
            self.__data_folder + "/rating_test.csv",
            header=0, names=columns
        )
        self.__test_data = self.__preprocess(self.__test_data)

    def __read_valset(self, columns):
        self.__val_data = pd.read_csv(
            self.__data_folder + "/rating_val.csv",
            header=0, names=columns
        )
        self.__val_data = self.__preprocess(self.__val_data)

    def __preprocess(self, data):
        """Map the id of all users and items according to user_dict and item_dict.
        To create the user_dict, all user ID in the training set is first sorted, then the first ID is map to 0 and so on.
        Do the same for item_dict.
        This process is done via `self.__read_trainset()`.

        Args:
            data (Dataframe): The dataset that need to be preprocessed.

        Returns:
            ndarray: The array with all id mapped.
        """
        # data['u_id'] = data['u_id'].replace(self.user_dict)
        # data['i_id'] = data['i_id'].replace(self.item_dict)

        data['u_id'] = data['u_id'].map(self.user_dict)
        data['i_id'] = data['i_id'].map(self.item_dict)

        # Tag unknown users/items with -1 (when val)
        data.fillna(-1, inplace=True)

        data['u_id'] = data['u_id'].astype(np.int32)
        data['i_id'] = data['i_id'].astype(np.int32)

        return data[['u_id', 'i_id', 'rating']].values

    def load_csv2ndarray(self, use_val=False, columns=['u_id', 'i_id', 'rating', 'timestamp']):
        """
        Load training set, validate set and test set via `.csv` file.
        Each as `ndarray`.

        Args:
            has_val (boolean): Denote if loading validate data or not. Defaults to True.
            columns (list): Columns name for DataFrame. Defaults to ['u_id', 'i_id', 'rating', 'timestamp'].

        Returns:
            train, val, test (np.array): Preprocessed data.
        """
        self.__read_trainset(columns)
        self.__read_testset(columns)

        if use_val:
            self.__read_valset(columns)
            return self.__train_data, self.__val_data, self.__test_data
        else:
            return self.__train_data, self.__test_data

    def load_genome_fromcsv(self, genome_file="genome_scores.csv", columns=["i_id", "g_id", "score"], reset_index=False):
        """
        Load genome scores from file.
        Args:
            genome_file (string): File name that contain genome scores. Must be in csv format.
            columns (list, optional): Columns name for DataFrame. Must be ["i_id", "g_id", "score"] or ["i_id", "score", "g_id"].
            reset_index (boolean): Reset the genome_tag column or not. Defaults to False.

        Returns:
            scores (DataFrame)
        """
        genome = pd.read_csv(
            self.__genome_folder + "/" + genome_file,
            header=0, names=columns
        )

        if reset_index:
            tag_map = {genome.g_id: newIdx for newIdx, genome in genome.loc[genome.i_id == 1].iterrows()}
            genome["g_id"] = genome["g_id"].map(tag_map)

        genome['i_id'] = genome['i_id'].map(self.item_dict)
        genome.fillna(0, inplace=True)

        return sparse.csr_matrix((genome['score'], (genome['i_id'].astype(int), genome['g_id'].astype(int)))).toarray()
