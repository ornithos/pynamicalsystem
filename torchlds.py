import torch
from torch.autograd import Variable
import math
import numpy as np
import time
import numpylds

def forward_lds_torch(y, A, C, Q, R, x0mu, x0sigma, as_log=True, validation=True, collect=True):

    def is_floatvar(x): return isinstance(x, Variable) and isinstance(x.data, torch.FloatTensor)
    assert is_floatvar(y) and y.ndimension() == 2, "y must be 2D torch.FloatTensor Variable"
    [T,n] = y.size()


    if validation:
        assert is_floatvar(A) and A.ndimension() == 2 and A.size()[0] == A.size()[1], "A must be square torch.FloatTensor Variable"
        assert is_floatvar(C) and C.ndimension() == 2 and C.size()[1] == A.size()[1] \
            and C.size()[0] == n, "C must be n x d torch.FloatTensor"
        d = C.size()[1]
        assert is_floatvar(Q) and Q.ndimension() == 2 and all([x==d for x in Q.size()]), "Q must be torch.FloatTensor conformable to A"
        assert is_floatvar(R) and R.ndimension() == 2 and all([x==n for x in R.size()]), "R must be n x n torch.FloatTensor."
        if not x0mu:
            x0mu = torch.zeros(d).float()
        assert isinstance(x0mu, torch.FloatTensor) and x0mu.ndimension() == 1 and x0mu.size()[0] == d, "x0mu must be a 1D torch vector conformable to A."
        # accept either scalar or matrix for x0sigma
        if isinstance(x0sigma, torch.FloatTensor):
            assert  x0sigma.ndimension() == 2 and all([x==d for x in x0sigma.size()]), "x0sigma must be conformable to size of A."
        else:
            assert isinstance(x0sigma, int) or isinstance(x0sigma, float), "x0sigma must be torch.FloatTensor or base scalar"
            x0sigma = torch.from_numpy(x0sigma * np.eye(d)).float()

    def gauss_logpdf(sigma, obs):
        # Log pdf of N(y|0,S).
        # if use builtin, may cause problems upstream in autodiff. Better to roll your own. Tested vs. scipy.
        CONST_Z       = - 0.5 * n * math.log(2 * math.pi)  # constant part of partition function
        cholS         = torch.potrf(sigma) # (Cholesky LAPACK) upper=True by default
        expon_vec, LU = torch.gesv(obs[:,None], cholS.t()) # exponent vector s.t. full exponent is self inner prod
        return -0.5 * torch.sum(expon_vec ** 2) - torch.sum(torch.log(torch.diag(cholS))) + CONST_Z


    # define Gauss posterior vars to hold cur value of p(x_t|y_1:t).
    m = Variable(x0mu, requires_grad=False)     # necessary in order to use '@' instead of torch.mm
    P = Variable(x0sigma, requires_grad=False)
    log_prob_accum = Variable(torch.zeros(1), requires_grad=True)
    log_probs = torch.zeros(T).float()

    if collect:
        filterMu = torch.zeros((T, d))
        filterSigma = torch.zeros((T, d, d))

    for tt in range(T):

        # forward predictive distribution
        m_minus = A @ m
        P_minus = A @ P @ A.t() + Q

        # form joint with y_{t+1}
        m_y    = C @ m_minus
        S      = C @ P_minus @ C.t() + R
        cov_xy = P_minus @ C.t()

        # intermediate quantities for conditioning (must be stable)
        S      = (S + S.t())/2
        K, LU  = torch.gesv(cov_xy.t(), S)  # covxy * inv(S). linsolve/gesv actual solves the commuted version. So t().
        K      = K.t()     # above returns the transpose of what we want. (Note that S.t() was not reqd. as symmetric.)

        delta_y= (y[tt,:] - m_y)
        m      = m_minus + K @ delta_y
        P      = P_minus - K @ S @ K.t()

        # collect log prob
        log_prob_tt    = gauss_logpdf(S, delta_y)
        log_prob_accum = log_prob_accum + log_prob_tt

        if collect:
            filterMu[tt,:] = m
            filterSigma[tt,:,:] = P
            log_probs[tt] = log_prob_tt.data[0]  # primarily for debugging purposes.


    loss = -log_prob_accum  # change objective to *minimise* not MLE.
    if not as_log:
        loss = torch.exp(loss)
    if collect:
        return (loss, log_probs, filterMu, filterSigma)
    else:
        return loss


def initialise_pars_torch():
    y, A, C, Q, R, x0sigma = numpylds.initialise_pars_numpy()
    y = Variable(torch.from_numpy(y).float(), requires_grad=False)
    A = Variable(torch.from_numpy(np.atleast_1d(A)).float(), requires_grad=True)
    C = Variable(torch.from_numpy(np.atleast_1d(C)).float(), requires_grad=True)
    Q = Variable(torch.from_numpy(np.atleast_1d(Q)).float(), requires_grad=True)
    R = Variable(torch.from_numpy(np.atleast_1d(R)).float(), requires_grad=True)
    if not isinstance(x0sigma, float) and not isinstance(x0sigma, int):
        x0sigma = torch.from_numpy(np.atleast_1d(x0sigma)).float()
    return y, A, C, Q, R, x0sigma


if __name__ == "__main__":
    y, A, C, Q, R, x0sigma = initialise_pars_torch()

    epochs = 200
    learning_rate = 1e-3

    for tt in range(epochs):
        loss = forward_lds_torch(y, A, C, Q, R, [], x0sigma, collect=False)
        print("Step {0:03d}: {1:.2f}".format(tt, loss.data[0]))

        # Use autograd to compute the backward pass.
        loss.backward()
        A.data -= learning_rate * A.grad.data
        C.data -= learning_rate * C.grad.data

        # Manually zero the gradients after running the backward pass
        A.grad.data.zero_()
        C.grad.data.zero_()


    if False:
        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Variables it should update.
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for t in range(500):
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x)

            # Compute and print loss.
            loss = loss_fn(y_pred, y)
            print(t, loss.data[0])

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable weights
            # of the model)
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its parameters
            optimizer.step()