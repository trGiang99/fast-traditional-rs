import numpy as np
from numba import njit

from .sim_helper import _run_cosine_params, _calculate_cosine_similarity, _run_pearson_params, _calculate_pearson_similarity


@timer("Computing Cosine similarity matrix took ")
def _cosine(n_x, yr, min_support=1):
    """Compute the cosine similarity between all pairs of users (or items).
    Only **common** users (or items) are taken into account.
    The cosine similarity is defined as:
    """
    prods = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)

    for y_ratings in yr:
        prods, freq, sqi, sqj = \
            _run_cosine_params(prods, freq, sqi, sqj, y_ratings)

    sim = _calculate_cosine_similarity(prods, freq, sqi, sqj, n_x, min_support)

    return sim


@timer("Computing Pearson similarity matrix took ")
def _pcc(n_x, yr, min_support=1):
    """Compute the cosine similarity between all pairs of users (or items).
    Only **common** users (or items) are taken into account.
    The cosine similarity is defined as:
    """
    prods = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sqi = np.zeros((n_x, n_x), np.double)
    sqj = np.zeros((n_x, n_x), np.double)
    si = np.zeros((n_x, n_x), np.double)
    sj = np.zeros((n_x, n_x), np.double)

    for y_ratings in yr:
        prods, freq, sqi, sqj, si, sj = \
            _run_pearson_params(prods, freq, sqi, sqj, si, sj, y_ratings)

    sim = _calculate_pearson_similarity(prods, freq, sqi, sqj, si, sj, n_x, min_support)

    return sim


@njit
def _cosine_genome(genome):
    """Calculate cosine simularity score between each movie
    using movie genome provided by MovieLens20M dataset.

    Args:
        genome (ndarray): movie genome, where each row contains genome score for that movie.

    Returns:
        S (ndarray): Similarity matrix
    """
    S = np.zeros((genome.shape[0], genome.shape[0]))

    for uidx in range(genome.shape[0]):
        for vidx in range(genome[(uidx+1):].shape[0]):
            numerator = genome[uidx].dot(genome[vidx+uidx+1])
            if (not numerator):
                S[uidx,vidx+uidx+1] = 0
                continue

            norm2_ratings_u = norm_np(genome[uidx])
            norm2_ratings_v = norm_np(genome[vidx+uidx+1])

            S[uidx,vidx+uidx+1] = (numerator / (norm2_ratings_u * norm2_ratings_v))

    return S


def _pcc_genome(genome):
    """Calculate Pearson correlation coefficient (pcc) simularity score between each movie
    using movie genome provided by MovieLens20M dataset.

    Args:
        genome (ndarray): movie genome, where each row contains genome score for that movie.

    Returns:
        S (ndarray): Similarity matrix
    """
    # Subtract mean, to calculate Pearson similarity score
    genome -= np.mean(genome, axis=1, keepdims=True)

    return _cosine_genome(genome)