# Fast Recommender System on MovieLens 20M Dataset (working in progress)
Inspired by [gbolmler implementation of SVD using numba](https://github.com/gbolmier/funk-svd).
This repo contains reimplementation of kNN and common matrix factorization methods using [numba](https://github.com/numba/numba) library to accelarate `numpy` operations. Numba is a cool library and you need to give this a shot for future implementation using `numpy`.

## Dataset
In this repo, I'm using the Movielens 20M Dataset from [Kaggle](https://www.kaggle.com/grouplens/movielens-20m-dataset).

This is a big dataset.
In order to extract the dataset to get a smaller dataset, first you need to download MovieLens 20M and save it on your computer, for example, to `movielens20M` folder.
Then you need to create a folder `movilens-sample` for the new sampling dataset.

On `sample_movielens.py` you can change the parameter to your like.

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

## Benchmarks

Folder `/examples` contains test runs on MovieLens dataset.

Compare to [NicolasHug/Surprise](https://github.com/NicolasHug/Surprise), the runtime of kNNBaseline using Pearson similarity scores is much faster (817s compared to 3166s of Surprise on MovieLens 20M dataset).
