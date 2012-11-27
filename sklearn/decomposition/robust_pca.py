# Author: Wei LI <kuantkid@gmail.com>
#         (mostly translation, see implementation details)
# License: BSD style

import numpy as np
from scipy.sparse.linalg import svds

from ..base import BaseEstimator, TransformerMixin

def _pos(A):
    """Postive Part of the matrix"""
    return A * np.array(A>0, np.int)


class RobustPCA(BaseEstimator, TransformerMixin):
    """Robust Principle Components Analysis (RobustPCA)

    Factorize the original design matrix into a low rank data
    matrix plus a sparse error matrix

    Parameters
    ----------
    alpha : float,
        Sparsity controlling parameter. Higher values lead to sparser
        components.

    max_iter : int,
        Maximum number of iterations to perform.

    tol : float,
        Tolerance for the stopping condition.

    method : {'svt'}
        alm: augmented lagrange multiplier method

    A_init : array of shape (n_samples, n_features),
        Initial values for the loadings for warm restart scenarios.

    E_init : array of shape (n_samples, n_features),
        Initial values for the components for warm restart scenarios.

    verbose :
        Degree of verbosity of the printed output.

    random_state : int or RandomState
        Pseudo number generator state used for random sampling.

    Attributes
    ----------
    `components_` : array, [n_components, n_features]
        Sparse components extracted from the data.

    `error_` : array
        Vector of errors at each iteration.
    """
    def __init__(self, lam=1., tau=.9, delta=0.1,
            max_iter=1000, tol=1e-8, method='svt', A_init=None,
            E_init=None, verbose=False, random_state=None):
        self.lam = lam
        self.delta = delta
        self.max_iter = max_iter
        self.tau = tau
        self.tol = tol
        self.method = method
        self.A_init = A_init
        self.E_init = E_init
        self.verbose = verbose
        self.random_state = random_state


    def _alg_svt(self, X):
        A_rank = 0
        tau = self.tau
        lam = self.lam
        tol = self.tol
        delta = self.delta

        n_samples, n_features = X.shape
        Y = np.zeros(shape=(n_samples, n_features),dtype=np.float)
        A = np.zeros(shape=(n_samples, n_features),dtype=np.float)
        E = np.zeros(shape=(n_samples, n_features),dtype=np.float)

        for _iter in range(self.max_iter):
            U,S,V = svds(X, A_rank + 1, 'L')
            print U.shape, S.shape, V.shape
            A = np.dot(np.dot(U,np.diag(_pos(S - tau))),V)
            E[:] = np.sign(Y) * _pos(np.abs(Y) - lam * tau)
            M = X - A - E
            
            A_rank = np.sum(S > tau)
            E_card = np.sum(np.abs(E) > 0)

            Y = Y + delta * M
            if self.verbose:
                print "%02d: |A|_f: %f rank(A): %d" %(_iter, (A*A).sum(), A_rank)
                print "\t |E|_f: %f |E|_0 : %d" %((E*E).sum(), E_card)
                print "\t |D-A-E|_F" %((M*M).sum())

            if ((X-A-E)**2).sum()/(X**2).sum() < tol:
                converged = true
                break
        return A, E
        
