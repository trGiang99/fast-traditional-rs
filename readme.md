# Fast Recommender System on MovieLens 20M Dataset (working in progress)
Inspired by [gbolmler implementation of SVD using numba](https://github.com/gbolmier/funk-svd).
This repo contains reimplementation of kNN and common matrix factorization methods using [numba](https://github.com/numba/numba) library to accelarate `numpy` operations. Numba is a cool library and you need to give this a shot for future implementation using `numpy`.

## MovieLens Dataset
The algorithms in this repo are tested on the Movielens [20M Dataset](https://www.kaggle.com/grouplens/movielens-20m-dataset).

This is a big dataset.
In order to extract the dataset to get a smaller dataset, first you need to download MovieLens 20M and save it on your computer, for example, to `movielens20M` folder.
Then you need to create a folder `movilens-sample` for the new sampling dataset.

On `utils/sample_movielens.py` you can change the parameter to your like.

```python
if __name__ == "__main__":
    sample_movielens(
       "movielens20M",
       "movielens-sample",
       sample_size=1000
    )
```

where `"movielens20M"` is the folder contains MovieLens 20M Dataset, `"movielens-sample"` is the folder contains new extracted dataset.
Size of the extracted dataset can be changed via `sample_size`.

## Netflix Prize Dataset
The algorithms in this repo are also tested on the [Netflix Prize dataset](https://www.kaggle.com/netflix-inc/netflix-prize-data).
Published by Netflix, the dataset contains a training set of 100 million ratings, which includes a probe set of 1 million ratings.
However, the qualifying dataset has not been published anywhere (to my knowledge).

For that reason, the scipt in `utils/split_netflix_dataset.py` first uses the probe set as the validation set, then split the remaining ratings into training set and testing set.
The output contains 3 distinct files, `rating_train.csv`, `rating_test.csv`, `rating_val.csv` just like MovieLens 20M, and can be loaded into the algorithms using `utils/DataLoader`.


## Benchmarks

Folder `/examples` contains test runs on MovieLens dataset.

Compare to [NicolasHug/Surprise](https://github.com/NicolasHug/Surprise), the runtime of kNNBaseline using Pearson similarity scores is much faster (817s compared to 3166s of Surprise on MovieLens 20M dataset).
