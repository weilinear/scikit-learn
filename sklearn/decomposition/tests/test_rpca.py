import numpy as np
from scipy.sparse import csr_matrix

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_less, assert_greater

from ..robust_pca import RobustPCA


def test_robust_pca_dense_matrix(seed=36):
    """PCA on dense arrays"""
    # generate low-rank dataset
    rank = 10
    n_features = 100
    n_samples = 1000
    noise_ratio = 0.0005
    noise_magnitude = 4
    
    # basis matrix
    rng = np.random.RandomState(seed=seed)
    B = rng.randn(rank, n_features)

    # coeeficient matrix
    A = rng.randn(n_samples, rank)

    # sparse noise matrix
    N = rng.randn(n_samples, n_features) * noise_magnitude
    N.flat[np.array(np.floor(rng.random_sample(np.floor(1-noise_ratio)) * N.size),
                    dtype=np.int)] = 0.0

    # data matrix
    D = np.dot(A, B) + N

    from scipy.io import savemat
    savemat("/tmp/data.mat", {"D":D});

    rpca = RobustPCA(verbose = True, max_iter = 10000)
    A, E = rpca._alg_svt(D)
    print E
    
    




    
