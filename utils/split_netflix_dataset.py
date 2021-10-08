import os
import random
import csv
from utils import timer


@timer(text='\nSplit the Netflix Prize dataset took ')
def split_netflix_dataset(netflix_path, trainset_ratio=0.8):
    """Split the Netflix Prize dataset to `rating_train.csv`, `rating_val.csv` and `rating_test.csv`.
    The dataset can be acquired from https://www.kaggle.com/netflix-inc/netflix-prize-data
    The full dataset contains 4 `combined_data_<id>.txt` files, where <id> are from 1 to 4.
    The probe dataset are provided in the file "probe.txt" and will be used as the validating set `rating_val.csv`.
    For the remaining ratings, by default, 80% are splited into `rating_train.csv`, and 20% for the `rating_test.csv`.

    Args:
        netflix_path (string): Folder contains the original MovieLens Dataset
        trainset_ratio (float): the percentage of ratings (after split the validation set from `probe.txt`) that will be randomly assigned as the training set.
    """

    train_file = open(netflix_path + "/rating_train.csv", 'w', encoding="utf-8", newline='')
    test_file = open(netflix_path + "/rating_test.csv", 'w', encoding="utf-8", newline='')
    val_file = open(netflix_path + "/rating_val.csv", 'w', encoding="utf-8", newline='')

    trainfile_csvwriter = csv.writer(train_file)
    testfile_csvwriter = csv.writer(test_file)
    valfile_csvwriter = csv.writer(val_file)

    trainfile_csvwriter.writerow(['userId', 'movieId', 'rating', 'timestamp'])
    testfile_csvwriter.writerow(['userId', 'movieId', 'rating', 'timestamp'])
    valfile_csvwriter.writerow(['userId', 'movieId', 'rating', 'timestamp'])

    probe_file = open(netflix_path + "/probe.txt", 'r')
    probe_data_line = probe_file.readline()
    probe_movieId = probe_data_line.strip().split(':')[0]   # The first movie Id in probe file
    probe_data_line = probe_file.readline()
    probe_userId = probe_data_line.strip()   # The first user Id in probe file

    for i in range(1, 5):
        print(f"\nReading combined_data_{i}.txt ...")

        netflix_file = open(netflix_path + f"/combined_data_{i}.txt", 'r')

        for train_data_line in netflix_file:
            train_data_line = train_data_line.strip()
            if train_data_line[-1] == ':':
                movieId = train_data_line.split(':')[0]
                continue

            userId, rating, timestamp = train_data_line.split(',')

            # Check if the dataline is include in the probe set
            if userId==probe_userId and movieId==probe_movieId:
                valfile_csvwriter.writerow([userId, movieId, rating, timestamp])

                # Update probe_movieId (if needed) and probe_userId by readling next line
                probe_data_line = probe_file.readline()
                probe_data_line = probe_data_line.strip()
                if probe_data_line[-1] == ':':
                    probe_movieId = probe_data_line.split(':')[0]
                    probe_data_line = probe_file.readline()
                probe_userId = probe_data_line.strip()

                continue

            mask = random.random()  # Uniform distribution (0, 1)
            if mask < trainset_ratio:   # Assign 80% of lines to the training set
                trainfile_csvwriter.writerow([userId, movieId, rating, timestamp])
            else:   # Assign 20% of lines to the training set
                testfile_csvwriter.writerow([userId, movieId, rating, timestamp])

        netflix_file.close()

    train_file.close()
    test_file.close()
    val_file.close()
    probe_file.close()

    print("Done.")


if __name__ == "__main__":
    split_netflix_dataset("netflixprize")
