import numpy as np
from numba import njit


@njit
def _run_cosine_params(prods, freq, sqi, sqj, y_ratings):
    for xi, ri in y_ratings:
        xi = int(xi)
        for xj, rj in y_ratings:
            xj = int(xj)
            freq[xi, xj] += 1
            prods[xi, xj] += ri * rj
            sqi[xi, xj] += ri**2
            sqj[xi, xj] += rj**2

    return prods, freq, sqi, sqj


@njit
def _calculate_cosine_similarity(prods, freq, sqi, sqj, n_x, min_sprt):
    sim = np.zeros((n_x, n_x), np.double)

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum

            sim[xj, xi] = sim[xi, xj]

    return sim


@njit
def _run_pearson_params(prods, freq, sqi, sqj, si, sj, y_ratings):
    for xi, ri in y_ratings:
        xi = int(xi)
        for xj, rj in y_ratings:
            xj = int(xj)
            freq[xi, xj] += 1
            prods[xi, xj] += ri * rj
            sqi[xi, xj] += ri**2
            sqj[xi, xj] += rj**2
            si[xi, xj] += ri
            sj[xi, xj] += rj

    return prods, freq, sqi, sqj, si, sj


@njit
def _calculate_pearson_similarity(prods, freq, sqi, sqj, si, sj, n_x, min_sprt):
    sim = np.zeros((n_x, n_x), np.double)

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                n = freq[xi, xj]
                num = n * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
                denum = np.sqrt((n * sqi[xi, xj] - si[xi, xj]**2) *
                                (n * sqj[xi, xj] - sj[xi, xj]**2))
                if denum == 0:
                    sim[xi, xj] = 0
                else:
                    sim[xi, xj] = num / denum

            sim[xj, xi] = sim[xi, xj]

    return sim


@njit
def _run_pearson_baseline_params(global_mean_, x_biases_, y_biases_, prods, freq, sq_diff_i, sq_diff_j, y, y_ratings):
    partial_bias = global_mean_ + y_biases_[y]
    for xi, ri in y_ratings:
        xi = int(xi)
        for xj, rj in y_ratings:
            xj = int(xj)
            freq[xi, xj] += 1
            diff_i = (ri - (partial_bias + x_biases_[xi]))
            diff_j = (rj - (partial_bias + x_biases_[xj]))
            prods[xi, xj] += diff_i * diff_j
            sq_diff_i[xi, xj] += diff_i**2
            sq_diff_j[xi, xj] += diff_j**2

    return prods, freq, sq_diff_i, sq_diff_j


@njit
def _calculate_pearson_baseline_similarity(prods, freq, sq_diff_i, sq_diff_j, n_x, shrinkage, min_sprt):
    sim = np.zeros((n_x, n_x), np.double)

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                sim[xi, xj] = prods[xi, xj] / (np.sqrt(sq_diff_i[xi, xj] *
                                                       sq_diff_j[xi, xj]))
                # the shrinkage part
                sim[xi, xj] *= (freq[xi, xj] - 1) / (freq[xi, xj] - 1 +
                                                     shrinkage)

            sim[xj, xi] = sim[xi, xj]

    return sim