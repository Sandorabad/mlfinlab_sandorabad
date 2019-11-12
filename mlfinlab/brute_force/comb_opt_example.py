"""
# =============================================================================
# COMB_OPT_EXAMPLE.PY
# =============================================================================
# Code snippets from the second half of Chapter 21 of "Advances in Financial
# Machine Learning" by Marcos López de Prado. These functions construct and
# evaluate a numerical example showing how the global optimum can be
# found using a digital computer.
# =============================================================================
"""

# Imports.


# =====================================================
# SNIPPET 21.4. PRODUCE A RANDOM MATRIX OF A GIVEN RANK
def random_matrix_with_rank(n_samples, n_cols, rank, sigma=0, hom_noise=True):
    """
    Produce a random 'n_samples'-by-'n_cols' matrix 'X' with given rank,
    and apply a specified random noise.
    
    :param n_samples: (int) Number of rows to appear in the matrix.
    :param n_cols: (int) Number of column to appear in the matrix.
    :param rank: (int) The rank of the matrix.
    :param sigma: (float) The variance of the noise to be produced.
    :param hom_noise: (bool) Whether or not to add homoscedastic noise.
     If False, the added noise is heterscedastic.
    :return: (numpy.array) The array 'X'.
    """
    rng = np.random.RandomState()
    U, _, _ = np.linalg.svd(rng.randn(n_cols, n_cols))
    X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)
    if hom_noise:
        X += sigma * rng.randn(n_samples, n_cols)  # Adding homoscedastic noise.
    else:
        sigmas = sigma * (rng.rand(n_cols)+.5)  # Adding heteroscedastic noise.
        X += rng.randn(n_samples, n_cols) * sigmas
    return X
