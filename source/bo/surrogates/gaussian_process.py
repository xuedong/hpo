import numpy as np
from scipy.linalg import cholesky, solve
from collections import OrderedDict
from scipy.optimize import minimize


class GaussianProcess:
    def __init__(self, covfunc, optimize=False, usegrads=False, mprior=0):
        """Gaussian Process regressor class.

        :type covfunc: instance from a class of covfunc module
        :param covfunc: covariance function
        :type optimize: bool
        :param optimize: whether to perform covariance function hyperparameter optimization
        :type usegrads: bool
        :param usegrads: whether to use gradient information on hyperparameter optimization, only used
        if `optimize=True`
        :type mprior: float
        :param mprior: explicit value for the mean function of the prior Gaussian Process
        """
        self.covfunc = covfunc
        self.optimize = optimize
        self.usegrads = usegrads
        self.mprior = mprior
        self.x = None
        self.y = None
        self.nsamples = None
        self.alpha = None
        self.logp = None
        self.K = None
        self.L = None

    def get_cov_params(self):
        """Current covariance function hyperparameters.

        :return: the dictionary containing covariance function hyperparameters
        """
        d = {}
        for param in self.covfunc.parameters:
            d[param] = self.covfunc.__dict__[param]
        return d

    def fit(self, x, y):
        """Fits a Gaussian Process regressor.

        :type x: np.ndarray, shape=(nsamples, nfeatures)
        :param x: training instances to fit the GP
        :type y: np.ndarray, shape=(nsamples,)
        :param y: corresponding continuous target values to x
        """
        self.x = x
        self.y = y
        self.nsamples = self.x.shape[0]
        if self.optimize:
            grads = None
            if self.usegrads:
                grads = self._grad
            self.opt_hyp(param_key=self.covfunc.parameters, param_bounds=self.covfunc.bounds, grads=grads)

        self.K = self.covfunc.cov(self.x, self.x)
        self.L = cholesky(self.K).T
        self.alpha = solve(self.L.T, solve(self.L, y - self.mprior))
        self.logp = -.5 * np.dot(self.y, self.alpha) - np.sum(np.log(np.diag(self.L))) - self.nsamples / 2 * np.log(
            2 * np.pi)

    def param_grad(self, k_param):
        """Gradient over hyperparameters. It is recommended to use `self._grad` instead.

        :type k_param: dict
        :param k_param: dictionary with keys being hyperparameters and values their queried values
        :return: the gradient corresponding to each hyperparameters, order given by `k_param.keys()`
        """
        k_param_key = list(k_param.keys())
        covfunc = self.covfunc.__class__(**k_param)
        # n = self.x.shape[0]
        cov_matrix = covfunc.cov(self.x, self.x)
        lower_matrix = cholesky(cov_matrix).T
        alpha = solve(lower_matrix.T, solve(lower_matrix, self.y))
        inner = np.dot(np.atleast_2d(alpha).T, np.atleast_2d(alpha)) - np.linalg.inv(cov_matrix)
        grads = []
        for param in k_param_key:
            grad_matrix = covfunc.grad_matrix(self.x, self.x, param=param)
            grad_matrix = .5 * np.trace(np.dot(inner, grad_matrix))
            grads.append(grad_matrix)
        return np.array(grads)

    def _lmlik(self, param_vector, param_key):
        """Marginal negative log-likelihood for given covariance hyperparameters.

        :type param_vector: list
        :param param_vector: list of values corresponding to hyperparameters to query
        :type param_key: list
        :param param_key: list of hyperparameter strings corresponding to `param_vector`
        :return: the negative log-marginal likelihood for chosen hyperparameters
        """
        k_param = OrderedDict()
        for k, v in zip(param_key, param_vector):
            k_param[k] = v
        self.covfunc = self.covfunc.__class__(**k_param)

        # This fixes recursion
        original_opt = self.optimize
        original_grad = self.usegrads
        self.optimize = False
        self.usegrads = False

        self.fit(self.x, self.y)

        self.optimize = original_opt
        self.usegrads = original_grad
        return - self.logp

    def _grad(self, param_vector, param_key):
        """Gradient for each hyperparameter, evaluated at a given point.

        :type param_vector: list
        :param param_vector: list of values corresponding to hyperparameters to query
        :type param_key: list
        :param: list of hyperparameter strings corresponding to `param_vector`
        :return: the gradient for each evaluated hyperparameter
        """
        k_param = OrderedDict()
        for k, v in zip(param_key, param_vector):
            k_param[k] = v
        return - self.param_grad(k_param)

    def opt_hyp(self, param_key, param_bounds, grads=None, n_trials=5):
        """Optimizes the negative marginal log-likelihood for given hyperparameters and bounds.
        This is an empirical Bayes approach (or Type II maximum-likelihood).

        :type param_key: list
        :param param_key: list of hyperparameters to optimize
        :type param_bounds: list
        :param param_bounds: list containing tuples defining bounds for each hyperparameter to optimize over
        :param grads: gradient matrix
        :param n_trials: number of trials
        """
        xs = [[1, 1, 1]]
        fs = [self._lmlik(xs[0], param_key)]
        for trial in range(n_trials):
            x0 = []
            for param, bound in zip(param_key, param_bounds):
                x0.append(np.random.uniform(bound[0], bound[1], 1)[0])
            if grads is None:
                res = minimize(self._lmlik, x0=np.array(x0), args=param_key, method='L-BFGS-B', bounds=param_bounds)
            else:
                res = minimize(self._lmlik, x0=np.array(x0), args=param_key, method='L-BFGS-B', bounds=param_bounds,
                               jac=grads)
            xs.append(res.x)
            fs.append(res.fun)

        arg_min = np.argmin(fs)
        opt_param = xs[int(arg_min)]
        k_param = OrderedDict()
        for k, x in zip(param_key, opt_param):
            k_param[k] = x
        self.covfunc = self.covfunc.__class__(**k_param)

    def predict(self, x_star, return_std=False):
        """Mean and covariances for the posterior Gaussian Process.

        :type x_star: np.ndarray, shape=((nsamples, nfeatures))
        :param x_star: testing instances to predict
        :type return_std: bool
        :param return_std: whether to return the standard deviation of the posterior process
        :return: mean and covariance of the posterior process for testing instances
        """
        x_star = np.atleast_2d(x_star)
        k_star = self.covfunc.cov(self.x, x_star).T
        fmean = self.mprior + np.dot(k_star, self.alpha)
        v = solve(self.L, k_star.T)
        fcov = self.covfunc.cov(x_star, x_star) - np.dot(v.T, v)
        if return_std:
            fcov = np.diag(fcov)
        return fmean, fcov

    def update(self, x_new, y_new):
        """Updates the internal model with `x_new` and `y_new` instances.

        :type x_new: np.ndarray, shape=((m, nfeatures))
        :param x_new: new training instances to update the model with
        :type y_new: np.ndarray, shape=((m,))
        :param y_new: new training targets to update the model with
        """
        y = np.concatenate((self.y, y_new), axis=0)
        x = np.concatenate((self.x, x_new), axis=0)
        self.fit(x, y)
