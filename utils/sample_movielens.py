import csv
import numpy as np
from utils import timer


@timer(text='\nSampling took ')
def sample_movielens(movielens_path, movielens_sample_path, sample_size=10000, test_ratio=0.2, val_ratio=0.05, sample_genome=False):
    """Sampling the original MovieLens Dataset into a dataset contain only 10k row
    All movies that doesn't appear in the raing file will be remove from the original movies.csv
    Also the genome_tags and genome_score.

    Args:
        movielens_path (string): Folder contains the original MovieLens Dataset
        movielens_sample_path (string): Folder to save new 10k MovieLens Dataset
        test_ratio (float): Size of test set over dataset. Defaults to 0.2
        val_ratio (float): Size of validate set over dataset. Defaults to 0.05
    """

    # List contains all movie's IDs in the new dataset
    movies_list = []

    print("\nReading the ratings file...")
    with open(movielens_path + "/rating.csv", 'r') as movielens:
        reader = csv.reader(movielens)

        train_set = []
        test_set = []
        val_set = []

        mask = np.ones(sample_size)
        mask[:round(sample_size*test_ratio)] = 0     # 0 indicate test data
        mask[round(-sample_size*val_ratio):] = 2     # 2 indicate validate data
        np.random.shuffle(mask)

        for idx, rating in enumerate(reader):
            if mask[idx] == 1:
                train_set.append(rating)
                if rating[1] not in movies_list:
                    movies_list.append(rating[1])   # Append new movie's Id to the movie list
            elif mask[idx] == 0:
                test_set.append(rating)
            else:
                val_set.append(rating)

            # Copy only first 10,000 (sample_size) rows
            if (idx >= sample_size-1):
                break

    print("Writing sample ratings...")
    # Write ratings to new path
    with open(movielens_sample_path + "/rating_train.csv", 'w', encoding="utf-8", newline='') as train_file:
        writer = csv.writer(train_file)
        writer.writerows(train_set)
    with open(movielens_sample_path + "/rating_test.csv", 'w', encoding="utf-8", newline='') as test_file:
        writer = csv.writer(test_file)
        writer.writerow(['userId', 'movieId', 'rating', 'timestamp'])
        writer.writerows(test_set)
    with open(movielens_sample_path + "/rating_val.csv", 'w', encoding="utf-8", newline='') as val_file:
        writer = csv.writer(val_file)
        writer.writerow(['userId', 'movieId', 'rating', 'timestamp'])
        writer.writerows(val_set)
    print("Done.")

    if sample_genome:
        print("\nReading genome tags file...")
        genome_scores = []
        # Get all genome score for the Ids in the movie_list above
        with open(movielens_path + "/genome_scores.csv", "r", encoding="utf-8") as scores:
            reader = csv.reader(scores)
            genome_scores = [score for _, score in enumerate(reader) if score[0] in movies_list]

        print("Writing genome tags...")
        # Write all needed score to new path
        with open(movielens_sample_path + "/genome_scores.csv", 'w', encoding="utf-8", newline='') as scores:
            writer = csv.writer(scores)
            writer.writerows(genome_scores)
        print("Done.")

    # print("Reading movies information...")
    # # Get information of all movies and assign it to the Ids in movie_list respectively
    # with open(movielens_path + "/movie.csv", 'r', encoding="utf-8") as movies:
    #     reader = csv.reader(movies)
    #     movies_list = [movie for _, movie in enumerate(reader) if movie[0] in movies_list]

    # print("Writing movies information...")
    # # Write all movies in the movie list to new path
    # with open(movielens_sample_path + "/movie.csv", 'w', encoding="utf-8", newline='') as movies:
    #     writer = csv.writer(movies)
    #     writer.writerows(movies_list)


if __name__ == "__main__":

    # Sampling MovieLens 20M Dataset to MovieLens 10k Dataset for the sake of testing
    sample_movielens("movielens20M", "movielens100k", sample_size=100000, sample_genome=False)