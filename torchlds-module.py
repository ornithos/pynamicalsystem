import torch
from torch.autograd import Variable
from torch import nn
import math
import numpy as np
import time
import numpylds
from types import SimpleNamespace
from functools import partial

class LGSSM(nn.Module):
    def __init__(self, y, latent_d):
        super(LGSSM, self).__init__()

        # process y
        assert isinstance(y, Variable) and isinstance(y.data, torch.FloatTensor) and y.ndimension() == 2, \
            "y must be 2D torch.FloatTensor Variable"
        [T, emission_d] = y.size()
        self.dims = SimpleNamespace()
        self.dims.T = T; self.dims.d = latent_d; self.dims.n = emission_d

        # define parameters
        self.A = torch.nn.Parameter(torch.FloatTensor(latent_d, latent_d).fill_(0))
        self.C = torch.nn.Parameter(torch.FloatTensor(emission_d, latent_d).fill_(0))
        self.Q = torch.nn.Parameter(torch.FloatTensor(latent_d, latent_d).fill_(0))
        self.R = torch.nn.Parameter(torch.FloatTensor(emission_d, emission_d).fill_(0))

        # define constants for x0
        self.x0mu    = torch.zeros(latent_d).float()
        self.x0sigma = torch.eye(latent_d, latent_d).float()
        self.cur_llh = torch.FloatTensor(1).fill_(float('-inf'))  # persist for retrieval after forward step


    def forward(self, debug=True):
        # define Gauss posterior vars to hold cur value of p(x_t|y_1:t).
        m = Variable(self.x0mu, requires_grad=False)  # necessary in order to use '@' instead of torch.mm
        P = Variable(self.x0sigma, requires_grad=False)
        log_prob_accum = Variable(torch.zeros(1), requires_grad=True)

        if debug:
            filterMu = torch.zeros((self.dims.T, self.dims.d))
            filterSigma = torch.zeros((self.dims.T, self.dims.d, self.dims.d))
            log_probs = torch.zeros(self.dims.T).float()

        # aliases for nicer looking math
        A = self.A; C = self.C; Q = self.Q; R = self.R

        for tt in range(self.dims.T):

            # forward predictive distribution
            m_minus = A @ m
            P_minus = A @ P @ A.t() + Q

            # form joint with y_{t+1}
            m_y = C @ m_minus
            S = C @ P_minus @ C.t() + R
            cov_xy = P_minus @ C.t()

            # intermediate quantities for conditioning (must be stable)
            S = (S + S.t()) / 2
            K, LU = torch.gesv(cov_xy.t(),S)  # covxy * inv(S). linsolve/gesv actual solves the commuted version. So t().
            K = K.t()  # above returns the transpose of what we want. (Note that S.t() was not reqd. as symmetric.)

            delta_y = (y[tt, :] - m_y)
            m = m_minus + K @ delta_y
            P = P_minus - K @ S @ K.t()

            # collect log prob
            log_prob_tt    = LGSSM.gauss_logpdf(S, delta_y)
            log_prob_accum = log_prob_accum + log_prob_tt

            if debug:
                filterMu[tt, :] = m.data
                filterSigma[tt, :, :] = P.data
                log_probs[tt] = log_prob_tt.data[0]

        if debug:
            return log_prob_accum, (log_probs, filterMu, filterSigma)

        self.cur_llh = log_prob_accum.data[0]
        return log_prob_accum




    def load_params_from_file(self, file):
        self.load_state_dict(torch.load(file))

    def set_parameter_values(self, A=None, C=None, Q=None, R=None, x0mu=None, x0sigma=None, validation=True):
        if validation:
            if A is not None: LGSSM.validate_input(A, 'A', (self.dims.d, ) * 2, isvar=True)
            if C is not None: LGSSM.validate_input(C, 'C', (self.dims.n,self.dims.d), isvar=True)
            if Q is not None: LGSSM.validate_input(Q, 'Q', (self.dims.d,) * 2, isvar=True)
            if R is not None: LGSSM.validate_input(R, 'R', (self.dims.n,) * 2, isvar=True)
            if x0mu is not None: LGSSM.validate_input(x0mu, 'x0mu', (self.dims.d,), isvar=False)
            if x0sigma is not None:
                if isinstance(x0sigma, int) or isinstance(x0sigma, float):
                    x0sigma = x0sigma * torch.eye(self.dims.d, self.dims.d).float()
                LGSSM.validate_input(x0sigma, 'x0sigma', (self.dims.d,) * 2, isvar=False)

        if A is not None: self.A.data = A.data
        if C is not None: self.C.data = C.data
        if Q is not None: self.Q.data = Q.data
        if R is not None: self.R.data = R.data
        if x0mu is not None: self.x0mu = x0mu
        if x0sigma is not None: self.x0sigma = x0sigma

    @staticmethod
    def validate_input(x, name, dims, isvar=True, dtype=torch.FloatTensor):
        assert isinstance(dims, tuple), "static method 'validate_input' used wrong. dims must be tuple."
        if isvar:
            assert isinstance(x, Variable), "{:s} is not a torch Variable.".format(name)
            assert isinstance(x.data, dtype), "{:s} should be wrapping {:s}. Got {:s}.".format(name,
                str(torch.FloatTensor), str(x.data.__class__))
        else:
            assert isinstance(x, dtype), "{:s} should be of type {:s}. Got {:s}.".format(name,
                str(torch.FloatTensor), str(x.__class__))
        shape = x.size()
        assert x.ndimension() == len(dims), "{:s} has {:d} dimensions. Expecting {:d}".format(name,
                x.ndimension(), len(dims))
        assert all([d1 == d2 for d1, d2 in zip(shape, dims)]), "{:s} of size {:s}. Expecting  {:s}." \
                .format(name, shape, dims)

    @staticmethod
    def gauss_logpdf(sigma, obs):
        # Log pdf of N(y|0,S).
        CONST_Z       = - 0.5 * obs.size()[0] * math.log(2 * math.pi)  # constant part of partition function
        cholS         = torch.potrf(sigma) # (Cholesky LAPACK) upper=True by default
        expon_vec, LU = torch.gesv(obs[:,None], cholS.t()) # exponent vector s.t. full exponent is self inner prod
        return -0.5 * torch.sum(expon_vec ** 2) - torch.sum(torch.log(torch.diag(cholS))) + CONST_Z


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
    latent_dim = 4
    opts = SimpleNamespace()
    opts.lr = 2e-3
    opts.beta1 = 0.5

    model_lds = LGSSM(y, latent_dim)
    model_lds.set_parameter_values(A=A, C=C, Q=Q, R=R, x0sigma=x0sigma)
    optimizer = torch.optim.LBFGS(model_lds.parameters(), max_iter=100)


    def closure(iter):
        optimizer.zero_grad()
        llh, _ = model_lds()
        objective = - llh
        print("step {0:03d}: {1:.2f}".format(iter, objective.data[0]))
        objective.backward()
        return objective
    optimizer.step(partial(closure, 0))

    optimizer = torch.optim.Adam(model_lds.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))

    for tt in range(epochs):
        # only need closure for some optimizers that re-evaluate fn (CG, LBFGS, ..?)
        optimizer.step(partial(closure, tt))
        #print("Step {0:03d}: {1:.2f}".format(tt, model_lds.cur_llh))

