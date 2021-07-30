import numpy as np
from utils import timer
from sklearn.metrics.pairwise import cosine_similarity

from .sim_helper import _run_cosine_params, _calculate_cosine_similarity, _run_pearson_params, _calculate_pearson_similarity, _run_pearson_baseline_params, _calculate_pearson_baseline_similarity


@timer("Computing Cosine similarity matrix took ")
def _cosine(n_x, yr, min_support=1):
    """Compute the cosine similarity between all pairs of users (or items).
    Only **common** users (or items) are taken into account.
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
    """Compute the Pearson coefficient correlation between all pairs of users (or items).
    Only **common** users (or items) are taken into account.
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


@timer("Computing Pearson Baseline similarity matrix took ")
def _pcc_baseline(n_x, yr, global_mean, bx, by, shrinkage=100, min_support=1):
    """Compute the Pearson Baseline coefficient correlation between all pairs of users (or items).
    Only **common** users (or items) are taken into account.
    """
    prods = np.zeros((n_x, n_x), np.double)
    freq = np.zeros((n_x, n_x), np.int)
    sq_diff_i = np.zeros((n_x, n_x), np.double)
    sq_diff_j = np.zeros((n_x, n_x), np.double)

    # Need this because of shrinkage. Pearson coeff is zero when support is 1, so that's OK.
    min_sprt = max(2, min_support)

    for y, y_ratings in enumerate(yr):
        prods, freq, sq_diff_i, sq_diff_j = \
            _run_pearson_baseline_params(global_mean, bx, by, prods, freq, sq_diff_i, sq_diff_j, y, y_ratings)

    sim = _calculate_pearson_baseline_similarity(prods, freq, sq_diff_i, sq_diff_j, n_x, shrinkage, min_support)

    return sim


@timer("Computing Cosine similarity for Tag Genome matrix took ")
def _cosine_genome(genome):
    """Calculate cosine simularity score between each movie
    using movie genome provided by MovieLens20M dataset.

    Args:
        genome (ndarray): movie genome, where each row contains genome score for that movie.

    Returns:
        S (ndarray): Similarity matrix
    """
    return cosine_similarity(genome, genome)


@timer("Computing Pearson similarity for Tag Genome matrix took ")
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