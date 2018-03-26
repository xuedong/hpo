from collections import OrderedDict

import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed

from bo.logger import EventLogger


class BO:
    def __init__(self, surrogate, acquisition, f, parameter_dict, n_jobs=1):
        """Bayesian Optimization class.

        :type surrogate: surrogate model instance
        :param surrogate: Gaussian Process surrogate model instance
        :type acquisition: acquisition instance
        :param acquisition: acquisition function instance
        :param f: target unction to maximize over parameters specified by `parameter_dict`
        :type parameter_dict: dict
        :param parameter_dict: dictionary specifying parameter, their type and bounds
        :type n_jobs: int
        :param n_jobs: parallel threads to use during acquisition optimization
        """
        self.GP = surrogate
        self.A = acquisition
        self.f = f
        self.parameters = parameter_dict
        self.n_jobs = n_jobs

        self.tau = None
        self.init_evals = None

        self.parameter_key = list(parameter_dict.keys())
        self.parameter_value = list(parameter_dict.values())
        self.parameter_type = [p[0] for p in self.parameter_value]
        self.parameter_range = [p[1] for p in self.parameter_value]

        self.history = []
        self.logger = EventLogger(self)

    def sample_param(self):
        """Randomly samples parameters over bounds.

        :return a random sample of specified parameters
        """
        d = OrderedDict()
        for index, param in enumerate(self.parameter_key):
            if self.parameter_type[index] == 'int':
                d[param] = np.random.randint(self.parameter_range[index][0], self.parameter_range[index][1])
            elif self.parameter_type[index] == 'cont':
                d[param] = np.random.uniform(self.parameter_range[index][0], self.parameter_range[index][1])
            else:
                raise ValueError('Unsupported variable type.')
        return d

    def _first_run(self, n_eval=3):
        """Performs initial evaluations before fitting GP.

        :return: the number of initial evaluations to perform, default is 3
        """
        self.X = np.empty((n_eval, len(self.parameter_key)))
        self.y = np.empty((n_eval,))
        for i in range(n_eval):
            s_param = self.sample_param()
            s_param_val = list(s_param.values())
            self.X[i] = s_param_val
            self.y[i] = self.f(**s_param)
        self.GP.fit(self.X, self.y)
        self.tau = np.max(self.y)
        self.history.append(self.tau)

    def _acq_wrapper(self, x_new):
        """Evaluates the acquisition function on a point

        :type x_new: np.ndarray, shape=((len(self.parameter_key),))
        :param x_new: point to evaluate the acquisition function on
        :return: the acquisition function value for `x_new`
        """
        new_mean, new_var = self.GP.predict(x_new, return_std=True)
        new_std = np.sqrt(new_var + 1e-6)
        return -self.A.eval(self.tau, new_mean, new_std)

    def _optimize_acq(self, method='L-BFGS-B', n_start=100):
        """Optimizes the acquisition function using a multi-start approach.

        :type method: str
        :param method: any `scipy.optimize` method that admits bounds and gradients, default is 'L-BFGS-B'
        :type n_start: int
        :param n_start: number of starting points for the optimization procedure, default is 100
        """
        start_points_dict = [self.sample_param() for _ in range(n_start)]
        start_points_arr = np.array([list(s.values()) for s in start_points_dict])
        x_best = np.empty((n_start, len(self.parameter_key)))
        f_best = np.empty((n_start,))
        if self.n_jobs == 1:
            for index, start_point in enumerate(start_points_arr):
                res = minimize(self._acq_wrapper, x0=start_point, method=method,
                               bounds=self.parameter_range)
                x_best[index], f_best[index] = res.x, np.atleast_1d(res.fun)[0]
        else:
            opt = Parallel(n_jobs=self.n_jobs)(delayed(minimize)(self._acq_wrapper,
                                                                 x0=start_point,
                                                                 method='L-BFGS-B',
                                                                 bounds=self.parameter_range) for start_point in
                                               start_points_arr)
            x_best = np.array([res.x for res in opt])
            f_best = np.array([np.atleast_1d(res.fun)[0] for res in opt])

        self.best = x_best[np.argmin(f_best)]

    def update_gp(self):
        """Updates the internal model with the next acquired point and its evaluation.
        """
        kw = {param: self.best[i] for i, param in enumerate(self.parameter_key)}
        f_new = self.f(**kw)
        self.GP.update(np.atleast_2d(self.best), np.atleast_1d(f_new))
        self.tau = np.max(self.GP.y)
        self.history.append(self.tau)

    def get_result(self):
        """Prints best result in the Bayesian Optimization procedure.

        :type: OrderedDict
        :return: the point yielding best evaluation in the procedure
        :type: float
        :return: the best function evaluation
        """
        arg_tau = np.argmax(self.GP.y)
        opt_x = self.GP.x[arg_tau]
        res_d = OrderedDict()
        for i, key in enumerate(self.parameter_key):
            res_d[key] = opt_x[i]
        return res_d, self.tau

    def run(self, max_iter=10, init_evals=3, resume=False):
        """Runs the Bayesian Optimization procedure.

        :type max_iter: int
        :param max_iter: number of iterations to run, default is 10
        :type init_evals: int
        :param init_evals: initial function evaluations before fitting a GP, default is 3
        :type resume: bool
        :param resume: whether to resume the optimization procedure from the last evaluation, default is `False`
        """
        if not resume:
            self.init_evals = init_evals
            self._first_run(self.init_evals)
            self.logger.print_init(self)
        for iteration in range(max_iter):
            self._optimize_acq()
            self.update_gp()
            self.logger.print_current(self)
