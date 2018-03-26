import numpy as np
from scipy.stats import norm


class Acquisition:
    def __init__(self, mode, eps=1e-06, **params):
        """Acquisition function class.

        :type mode: str
        :param mode: the behaviour of the acquisition strategy, currently supported values are
        `expected_improvement`, `probability_improvement`, `gpucb`, `entropy`.
        :type eps: float
        :param eps: small floating value to avoid `np.sqrt` or zero-division warnings.
        """
        self.params = params
        self.eps = eps
        self.tau = None
        self.mean = None
        self.std = None

        mode_dict = {
            'expected_improvement': self.expected_improvement,
            'probability_improvement': self.probability_improvement,
            'gpucb': self.gpucb,
            'entropy': self.entropy
        }

        self.f = mode_dict[mode]

    def probability_improvement(self, tau, mean, std):
        """Probability of improvement acquisition function.

        :type tau: float
        :param tau: best observed function evaluation
        :type mean: float
        :param mean: point mean of the posterior process
        :type std: float
        :param std: point std of the posterior process
        :return: the probability of improvement
        """
        z = (mean - tau - self.eps) / (std + self.eps)
        return norm.cdf(z)

    def expected_improvement(self, tau, mean, std):
        """Expected Improvement acquisition function.

        :type tau: float
        :param tau: best observed function evaluation
        :type mean: float
        :param mean: point mean of the posterior process
        :type std: float
        :param std: point std of the posterior process
        :return: the expected improvement
        """
        z = (mean - tau - self.eps) / (std + self.eps)
        return (mean - tau) * norm.cdf(z) + std * norm.pdf(z)[0]

    def gpucb(self, mean, std, beta):
        """Upper-confidence bound acquisition function.

        :type mean: float
        :param mean: point mean of the posterior process
        :type std: float
        :param std: point std of the posterior process
        :type beta: float
        :param beta: constant
        :return: the upper-confidence bound
        """
        self.mean = mean  # unnecessary code to avoid code inspection
        return mean + beta * std

    def entropy(self, tau, mean, std, sigman):
        """Predictive entropy acquisition function.

        :type tau: float
        :param tau: best observed function evaluation
        :type mean: float
        :param mean: point mean of the posterior process
        :type std: float
        :param std: point std of the posterior process
        :type sigman: float
        :param sigman: noise variance
        :return: the predictive entropy
        """
        self.tau = tau
        self.mean = mean
        sp2 = std ** 2 + sigman
        return 0.5 * np.log(2 * np.pi * np.e * sp2)

    def eval(self, tau, mean, std):
        """Evaluates selected acquisition function.

        :type tau: float
        :param tau: best observed function evaluation
        :type mean: float
        :param mean: point mean of the posterior process
        :type std: float
        :param std: point std of the posterior process
        :return: acquisition function value
        """
        return self.f(tau, mean, std, **self.params)
