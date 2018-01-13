import numpy as np
from scipy.stats import multivariate_normal
from scipy.io import loadmat
import os
import time

def forward_lds_numpy(y, A, C, Q, R, x0mu, x0sigma, as_log=True, validation=True, collect=True):

    assert isinstance(y, np.ndarray) and y.ndim == 2, "y must be 2D numpy array"
    [T,n] = y.shape

    if validation:
        assert isinstance(A, np.ndarray) and A.ndim == 2 and A.shape[0] == A.shape[1], "A must be square numpy array"
        assert isinstance(C, np.ndarray) and C.ndim == 2 and C.shape[1] == A.shape[1] \
            and C.shape[0] == n, "C must be n x d numpy array"
        d = C.shape[1]
        assert isinstance(Q, np.ndarray) and Q.ndim == 2 and all([x==d for x in Q.shape]), "Q must be numpy array conformable to A"
        assert isinstance(R, np.ndarray) and R.ndim == 2 and all([x==n for x in R.shape]), "R must be n x n numpy array."
        if not x0mu:
            x0mu = np.zeros(d)
        assert isinstance(x0mu, np.ndarray) and x0mu.ndim == 1 and x0mu.shape[0] == d, "x0mu must be a 1D vector conformable to A."
        # accept either scalar or matrix for x0sigma
        if isinstance(x0sigma, np.ndarray):
            assert  x0sigma.ndim == 2 and all([x==d for x in x0sigma.shape]), "x0sigma must be conformable to size of A."
        else:
            assert isinstance(x0sigma, int) or isinstance(x0sigma, float)


    # define Gauss posterior vars to hold cur value of p(x_t|y_1:t).
    m = x0mu
    P = x0sigma
    log_probs = np.zeros(T)

    if collect:
        filterMu = np.zeros((T, d))
        filterSigma = np.zeros((T, d, d))

    for tt in range(T):

        # forward predictive distribution
        m_minus = A.dot(m)
        P_minus = A.dot(P).dot(A.T) + Q

        # form joint with y_{t+1}
        m_y    = C.dot(m_minus)
        S      = C.dot(P_minus).dot(C.T) + R
        cov_xy = P_minus.dot(C.T)

        # intermediate quantities for conditioning (must be stable)
        S      = (S + S.T)/2
        K      = np.linalg.solve(S.T, cov_xy.T).T # covxy * inv(S), but better. Transposes are ugly.

        delta_y= (y[tt,:] - m_y)
        m      = m_minus + K.dot(delta_y)
        P      = P_minus - K.dot(S).dot(K.T)

        # collect log prob
        # if use builtin, may cause problems upstream in autodiff. Better to roll your own. Tested vs. scipy.
        cholS         = np.linalg.cholesky(S)
        expon_vec     = np.linalg.solve(cholS, delta_y)
        log_probs[tt] =  -0.5 * np.sum(expon_vec ** 2) - np.sum(np.log(np.diag(cholS))) -0.5*n*np.log(2*np.pi)
        #log_probs[tt] = multivariate_normal.logpdf(delta_y, np.zeros(n), S)

        if collect:
            filterMu[tt,:] = m
            filterSigma[tt,:,:] = P


    probs = log_probs
    if not as_log:
        log_probs = np.exp(log_probs)
    if collect:
        return (probs, filterMu, filterSigma)
    else:
        return probs

def initialise_pars_numpy():
    A = np.array([[0.50, -0.30, -0.18, -0.22],
         [0.34, 0.15, 0.11, 0],
         [-0.24, 0.00, 0.30, 0.11],
         [0.08, 0.35, 0.24, 0.75]])
    C = np.array([[1, 0, 0, 1],
         [0, 1, 0, 1],
         [0, 0, 1, 1]])
    Q = 0.2 * np.eye(4)
    R = 0.002 * np.eye(3)

    C = C[[0,2],:]
    R = R[np.ix_([0,2],[0,2])]
    x0sigma = 1e-8

    matlab_data = loadmat(os.path.join("/home/alex/code/matlab", "gen_data18_01_12.mat"))
    matlab      = dict()
    for field in matlab_data['gen_data'][0].dtype.fields:
        matlab[field] = matlab_data['gen_data'][0][field][0]

    y = matlab['y'].T
    return y, A, C, Q, R, x0sigma

if __name__ == "__main__":

    y, A, C, Q, R, x0sigma = initialise_pars_numpy()
    lds_out = forward_lds_numpy(y, A, C, Q, R, [], x0sigma, collect=True)

    if False:
        pre_loop_time = time.time()
        for _ in range(1000):
            lds_out = forward_lds_numpy(y, A, C, Q, R, [], x0sigma, collect=False)
        print("Finished in {0:.4f} seconds".format(time.time() - pre_loop_time))