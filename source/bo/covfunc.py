import numpy as np
from scipy.special import gamma, kv
from scipy.spatial.distance import cdist

default_bounds = {
    'lscale': [1e-4, 1],
    'sigmaf': [1e-4, 2],
    'sigman': [1e-6, 2],
    'v': [1e-3, 10],
    'gamma': [1e-3, 1.99],
    'alpha': [1e-3, 1e4],
    'period': [1e-3, 10]
}


def l2_norm(x, x_star):
    """Wrapper function to compute the L2 norm.
 
    :type x: np.ndarray, shape=((n, nfeatures))
    :param x: instances
    :type x_star: np.ndarray, shape((m, nfeatures))
    :param x_star: instances
    :return: pairwise Euclidean distance between row pairs of `x` and `x_star`
    """
    return cdist(x, x_star)


def kronecker_delta(x, x_star):
    """Computes Kronecker delta for rows in x and x_star.

    :type x: np.ndarray, shape=((n, nfeatures))
    :param x: instances
    :type x_star: np.ndarray, shape((m, nfeatures))
    :param x_star: instances
    :return: Kronecker delta between row pairs of `x` and `x_star`
    """
    return cdist(x, x_star) < np.finfo(np.float32).eps


class SquaredExponential:
    def __init__(self, lscale=1, sigmaf=1.0, sigman=1e-6, bounds=None, parameters={'lscale', 'sigmaf', 'sigman'}):
        """Squared exponential kernel class.

        :type lscale: float
        :param lscale: characteristic length-scale, units in input space
        in which posterior GP values do not change significantly.
        :type sigmaf: float
        :param sigmaf: signal variance, controls the overall scale of the covariance function
        :type sigman: float
        :param sigman: noise variance, additive noise in output space
        :type bounds: list
        :param bounds: list of tuples specifying hyperparameter range in optimization procedure
        :type parameters: list
        :param parameters: list of strings specifying which hyperparameters should be optimized
        """
        self.lscale = lscale
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def cov(self, x, x_star):
        """Computes covariance function values over `x` and `x_star`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape=((n, nfeatures))
        :param x_star: instances
        :return: the computed covariance matrix
        """
        r = l2_norm(x, x_star)
        return self.sigmaf * np.exp(-.5 * r ** 2 / self.lscale ** 2) + self.sigman * kronecker_delta(x, x_star)

    def grad_matrix(self, x, x_star, param='lscale'):
        """Computes gradient matrix for instances `x`, `x_star` and hyperparameter `param`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param x_star: instances
        :type param: str
        :param param: parameter to compute gradient matrix for
        :return: gradient matrix for parameter `param`
        """
        if param == 'lscale':
            r = l2_norm(x, x_star)
            num = r ** 2 * self.sigmaf * np.exp(-r ** 2 / (2 * self.lscale ** 2))
            den = self.lscale ** 3
            l_grad = num / den
            return l_grad
        elif param == 'sigmaf':
            r = l2_norm(x, x_star)
            sigmaf_grad = (np.exp(-.5 * r ** 2 / self.lscale ** 2))
            return sigmaf_grad

        elif param == 'sigman':
            sigman_grad = kronecker_delta(x, x_star)
            return sigman_grad

        else:
            raise ValueError('Param not found')


class Matern:
    def __init__(self, v=1, lscale=1, sigmaf=1, sigman=1e-6, bounds=None,
                 parameters={'v', 'lscale', 'sigmaf', 'sigman'}):
        """Matern kernel class.

        :type v: float
        :param v: scale-mixture hyperparameter of the Matern covariance function
        :type lscale: float
        :param lscale: characteristic length-scale, units in input space
        in which posterior GP values do not change significantly.
        :type sigmaf: float
        :param sigmaf: signal variance, controls the overall scale of the covariance function
        :type sigman: float
        :param sigman: noise variance, additive noise in output space
        :type bounds: list
        :param bounds: list of tuples specifying hyperparameter range in optimization procedure
        :type parameters: list
        :param parameters: list of strings specifying which hyperparameters should be optimized
        """
        self.v, self.lscale = v, lscale
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def cov(self, x, x_star):
        """Computes covariance function values over `x` and `x_star`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param x_star: instances
        :return: the computed covariance matrix
        """
        r = l2_norm(x, x_star)
        bessel = kv(self.v, np.sqrt(2 * self.v) * r / self.lscale)
        f = 2 ** (1 - self.v) / gamma(self.v) * (np.sqrt(2 * self.v) * r / self.lscale) ** self.v
        res = f * bessel
        res[np.isnan(res)] = 1
        res = self.sigmaf * res + self.sigman * kronecker_delta(x, x_star)
        return res


class Matern32:
    def __init__(self, lscale=1, sigmaf=1, sigman=1e-6, bounds=None, parameters={'lscale', 'sigmaf', 'sigman'}):
        """Matern v=3/2 kernel class.

        :type lscale: float
        :param lscale: characteristic length-scale, units in input space
        in which posterior GP values do not change significantly.
        :type sigmaf: float
        :param sigmaf: signal variance, controls the overall scale of the covariance function
        :type sigman: float
        :param sigman: noise variance, additive noise in output space
        :type bounds: list
        :param bounds: list of tuples specifying hyperparameter range in optimization procedure
        :type parameters: list
        :param parameters: list of strings specifying which hyperparameters should be optimized
        """

        self.lscale = lscale
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def cov(self, x, x_star):
        """Computes covariance function values over `x` and `x_star`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param x_star: instances
        :return: the computed covariance matrix
        """
        r = l2_norm(x, x_star)
        one = (1 + np.sqrt(3 * (r / self.lscale) ** 2))
        two = np.exp(- np.sqrt(3 * (r / self.lscale) ** 2))
        return self.sigmaf * one * two + self.sigman * kronecker_delta(x, x_star)

    def grad_matrix(self, x, x_star, param):
        """Computes gradient matrix for instances `x`, `x_star` and hyperparameter `param`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param x_star: instances
        :type param: str
        :param param: parameter to compute gradient matrix for
        :return: the gradient matrix for parameter `param`
        """
        if param == 'lscale':
            r = l2_norm(x, x_star)
            num = 3 * (r ** 2) * self.sigmaf * np.exp(-np.sqrt(3) * r / self.lscale)
            return num / (self.lscale ** 3)
        elif param == 'sigmaf':
            r = l2_norm(x, x_star)
            one = (1 + np.sqrt(3 * (r / self.lscale) ** 2))
            two = np.exp(- np.sqrt(3 * (r / self.lscale) ** 2))
            return one * two
        elif param == 'sigman':
            return kronecker_delta(x, x_star)
        else:
            raise ValueError('Param not found')


class Matern52:
    def __init__(self, lscale=1, sigmaf=1, sigman=1e-6, bounds=None, parameters={'lscale', 'sigmaf', 'sigman'}):
        """Matern v=5/2 kernel class.

        :type lscale: float
        :param lscale: characteristic length-scale, units in input space
        in which posterior GP values do not change significantly.
        :type sigmaf: float
        :param sigmaf: signal variance, controls the overall scale of the covariance function
        :type sigman: float
        :param sigman: noise variance, additive noise in output space
        :type bounds: list
        :param bounds: list of tuples specifying hyperparameter range in optimization procedure
        :type parameters: list
        :param parameters: list of strings specifying which hyperparameters should be optimized
        """
        self.lscale = lscale
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def cov(self, x, x_star):
        """Computes covariance function values over `x` and `x_star`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param: instances
        :return: the computed covariance matrix
        """
        r = l2_norm(x, x_star)
        one = (1 + np.sqrt(5 * (r / self.lscale) ** 2) + 5 * (r / self.lscale) ** 2 / 3)
        two = np.exp(-np.sqrt(5 * r ** 2))
        return self.sigmaf * one * two + self.sigman * kronecker_delta(x, x_star)

    def grad_matrix(self, x, x_star, param):
        """Computes gradient matrix for instances `x`, `x_star` and hyperparameter `param`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param x_star: instances
        :type param: str
        :param param: parameter to compute gradient matrix for
        :return:the gradient matrix for parameter `param`
        """
        r = l2_norm(x, x_star)
        if param == 'lscale':
            num_one = 5 * r ** 2 * np.exp(-np.sqrt(5) * r / self.lscale)
            num_two = np.sqrt(5) * r / self.lscale + 1
            res = num_one * num_two / (3 * self.lscale ** 3)
            return res
        elif param == 'sigmaf':
            one = (1 + np.sqrt(5 * (r / self.lscale) ** 2) + 5 * (r / self.lscale) ** 2 / 3)
            two = np.exp(-np.sqrt(5 * r ** 2))
            return one * two
        elif param == 'sigman':
            return kronecker_delta(x, x_star)


class GammaExponential:
    def __init__(self, exp_gamma=1, lscale=1, sigmaf=1, sigman=1e-6, bounds=None,
                 parameters={'exp_gamma', 'lscale', 'sigmaf', 'sigman'}):
        """Gamma-exponential kernel class.

        :type exp_gamma: float
        :param exp_gamma: hyperparameter of the Gamma-exponential covariance function
        :type lscale: float
        :param lscale: characteristic length-scale, units in input space
        in which posterior GP values do not change significantly.
        :type sigmaf: float
        :param sigmaf: signal variance, controls the overall scale of the covariance function
        :type sigman: float
        :param sigman: noise variance, additive noise in output space
        :type bounds: list
        :param bounds: list of tuples specifying hyperparameter range in optimization procedure
        :type parameters: list
        :param parameters: list of strings specifying which hyperparameters should be optimized
        """
        self.gamma = exp_gamma
        self.lscale = lscale
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def cov(self, x, x_star):
        """Computes covariance function values over `x` and `x_star`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param x_star: instances
        :return: the computed covariance matrix
        """
        r = l2_norm(x, x_star)
        return self.sigmaf * (np.exp(-(r / self.lscale) ** self.gamma)) + self.sigman * kronecker_delta(x, x_star)

    def grad_matrix(self, x, x_star, param):
        """Computes gradient matrix for instances `x`, `x_star` and hyperparameter `param`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param x_star: instances
        :type param: str
        :param param: parameter to compute gradient matrix for.
        :return: the gradient matrix for parameter `param`
        """
        if param == 'gamma':
            eps = 10e-6
            r = l2_norm(x, x_star) + eps
            first = -np.exp(- (r / self.lscale) ** self.gamma)
            sec = (r / self.lscale) ** self.gamma * np.log(r / self.lscale)
            gamma_grad = first * sec
            return gamma_grad
        elif param == 'lscale':
            r = l2_norm(x, x_star)
            num = self.gamma * np.exp(-(r / self.lscale) ** self.gamma) * (r / self.lscale) ** self.gamma
            l_grad = num / self.lscale
            return l_grad
        elif param == 'sigmaf':
            r = l2_norm(x, x_star)
            sigmaf_grad = (np.exp(-(r / self.lscale) ** self.gamma))
            return sigmaf_grad
        elif param == 'sigman':
            sigman_grad = kronecker_delta(x, x_star)
            return sigman_grad
        else:
            raise ValueError('Param not found')


class RationalQuadratic:
    def __init__(self, alpha=1, lscale=1, sigmaf=1, sigman=1e-6, bounds=None,
                 parameters={'alpha', 'lscale', 'sigmaf', 'sigman'}):
        """Rational-quadratic kernel class.

        :type alpha: float
        :param alpha: hyperparameter of the rational-quadratic covariance function
        :type lscale: float
        :param lscale: characteristic length-scale, units in input space
        in which posterior GP values do not change significantly.
        :type sigmaf: float
        :param sigmaf: signal variance, controls the overall scale of the covariance function
        :type sigman: float
        :param sigman: noise variance, additive noise in output space
        :type bounds: list
        :param bounds: list of tuples specifying hyperparameter range in optimization procedure
        :type parameters: list
        :param parameters: list of strings specifying which hyperparameters should be optimized
        """
        self.alpha = alpha
        self.lscale = lscale
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def cov(self, x, x_star):
        """Computes covariance function values over `x` and `x_star`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param x_star: instances
        :return: the computed covariance matrix
        """
        r = l2_norm(x, x_star)
        return self.sigmaf * ((1 + r ** 2 / (2 * self.alpha * self.lscale ** 2))
                              ** (-self.alpha)) + self.sigman * kronecker_delta(x, x_star)

    def grad_matrix(self, x, x_star, param):
        """Computes gradient matrix for instances `x`, `x_star` and hyperparameter `param`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param x_star: instances
        :type param: str
        :param param: parameter to compute gradient matrix for
        :return: the gradient matrix for parameter `param`
        """
        if param == 'alpha':
            r = l2_norm(x, x_star)
            one = (r ** 2 / (2 * self.alpha * self.lscale ** 2) + 1) ** (-self.alpha)
            two = r ** 2 / ((2 * self.alpha * self.lscale ** 2) * (r ** 2 / (2 * self.alpha * self.lscale ** 2) + 1))
            three = np.log(r ** 2 / (2 * self.alpha * self.lscale ** 2) + 1)
            alpha_grad = one * (two - three)
            return alpha_grad
        elif param == 'lscale':
            r = l2_norm(x, x_star)
            num = r ** 2 * (r ** 2 / (2 * self.alpha * self.lscale ** 2) + 1) ** (-self.alpha - 1)
            l_grad = num / self.lscale ** 3
            return l_grad
        elif param == 'sigmaf':
            r = l2_norm(x, x_star)
            sigmaf_grad = (1 + r ** 2 / (2 * self.alpha * self.lscale ** 2)) ** (-self.alpha)
            return sigmaf_grad
        elif param == 'sigman':
            sigman_grad = kronecker_delta(x, x_star)
            return sigman_grad
        else:
            raise ValueError('Param not found')


class ExpSine:
    def __init__(self, lscale=1.0, period=1.0, bounds=None, parameters={'lscale', 'period'}):
        """Exponential sine kernel class.

        :type lscale: float
        :param lscale: characteristic length-scale, units in input space
        in which posterior GP values do not change significantly.
        :type period: float
        :param period: period hyperparameter
        :type bounds: list
        :param bounds: list of tuples specifying hyperparameter range in optimization procedure
        :type parameters: list
        :param parameters: list of strings specifying which hyperparameters should be optimized
        """
        self.period = period
        self.lscale = lscale
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def cov(self, x, x_star):
        """Computes covariance function values over `x` and `x_star`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param x_star: instances
        :return: the computed covariance matrix
        """
        r = l2_norm(x, x_star)
        num = - 2 * np.sin(np.pi * r / self.period)
        return np.exp(num / self.lscale) ** 2 + 1e-4

    def grad_matrix(self, x, x_star, param):
        """Computes gradient matrix for instances `x`, `x_star` and hyperparameter `param`.
        
        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param: instances 
        :param param: 
        :return: the gradient matrix for parameter `param`
        """
        if param == 'lscale':
            r = l2_norm(x, x_star)
            one = 4 * np.sin(np.pi * r / self.period)
            two = np.exp(-4 * np.sin(np.pi * r / self.period) / self.lscale)
            return one * two / (self.lscale ** 2)
        elif param == 'period':
            r = l2_norm(x, x_star)
            one = 4 * np.pi * r * np.cos(np.pi * r / self.period)
            two = np.exp(-4 * np.sin(np.pi * r / self.period) / self.lscale)
            return one * two / (self.lscale * self.period ** 2)


class DotProd:
    def __init__(self, sigmaf=1.0, sigman=1e-6, bounds=None, parameters={'sigmaf', 'sigman'}):
        """Dot-product kernel class.

        :type sigmaf: float
        :param sigmaf: signal variance, controls the overall scale of the covariance function
        :type sigman: float
        :param sigman: noise variance, additive noise in output space
        :type bounds: list
        :param bounds: list of tuples specifying hyperparameter range in optimization procedure
        :type parameters: list
        :param parameters: list of strings specifying which hyperparameters should be optimized
        """
        self.sigmaf = sigmaf
        self.sigman = sigman
        self.parameters = parameters
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = []
            for param in self.parameters:
                self.bounds.append(default_bounds[param])

    def cov(self, x, x_star):
        """Computes covariance function values over `x` and `x_star`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param x_star: instances
        :return: the computed covariance matrix
        """
        return self.sigmaf * np.dot(x, x_star.T) + self.sigman * kronecker_delta(x, x_star)

    def grad_matrix(self, x, x_star, param):
        """Computes gradient matrix for instances `x`, `x_star` and hyperparameter `param`.

        :type x: np.ndarray, shape=((n, nfeatures))
        :param x: instances
        :type x_star: np.ndarray, shape((m, nfeatures))
        :param x_star: instances
        :type param: str
        :param param: parameter to compute gradient matrix for
        :return: the gradient matrix for parameter `param`
        """
        if param == 'sigmaf':
            return np.dot(x, x_star.T)
        elif param == 'sigman':
            return self.sigmaf * np.dot(x, x_star.T)
