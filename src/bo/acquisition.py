import numpy as np
from scipy.stats import norm, t


class Acquisition:
    def __init__(self, mode, eps=1e-06, **params):
        """
        Acquisition function class.

        Parameters
        ----------
        mode: str
            Defines the behaviour of the acquisition strategy. Currently supported values are
            `ExpectedImprovement`, `ProbabilityImprovement`, `UCB`, `Entropy`.
        eps: float
            Small floating value to avoid `np.sqrt` or zero-division warnings.
        params: float
            Extra parameters needed for certain acquisition functions, e.g. UCB needs
            to be supplied with `beta`.
        """
        self.params = params
        self.eps = eps

        mode_dict = {
            'ExpectedImprovement': self.ExpectedImprovement,
            'ProbabilityImprovement': self.ProbabilityImprovement,
            'UCB': self.UCB,
            'Entropy': self.Entropy
        }

        self.f = mode_dict[mode]

    def ProbabilityImprovement(self, tau, mean, std):
        """
        Probability of Improvement acquisition function.

        Parameters
        ----------
        tau: float
            Best observed function evaluation.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.

        Returns
        -------
        float
            Probability of improvement.
        """
        z = (mean - tau - self.eps) / (std + self.eps)
        return norm.cdf(z)

    def ExpectedImprovement(self, tau, mean, std):
        """
        Expected Improvement acquisition function.

        Parameters
        ----------
        tau: float
            Best observed function evaluation.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.

        Returns
        -------
        float
            Expected improvement.
        """
        z = (mean - tau - self.eps) / (std + self.eps)
        return (mean - tau) * norm.cdf(z) + std * norm.pdf(z)[0]

    def UCB(self, tau, mean, std, beta):
        """
        Upper-confidence bound acquisition function.

        Parameters
        ----------
        tau: float
            Best observed function evaluation.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.
        beta: float
            Hyperparameter controlling exploitation/exploration ratio.

        Returns
        -------
        float
            Upper confidence bound.
        """
        return mean + beta * std

    def Entropy(self, tau, mean, std, sigman):
        """
        Predictive entropy acquisition function

        Parameters
        ----------
        tau: float
            Best observed function evaluation.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.
        sigman: float
            Noise variance

        Returns
        -------
        float:
            Predictive entropy.
        """
        sp2 = std **2 + sigman
        return 0.5 * np.log(2 * np.pi * np.e * sp2)

    def eval(self, tau, mean, std):
        """
        Evaluates selected acquisition function.

        Parameters
        ----------
        tau: float
            Best observed function evaluation.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.

        Returns
        -------
        float
            Acquisition function value.

        """
        return self.f(tau, mean, std, **self.params)
