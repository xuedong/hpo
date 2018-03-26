import numpy as np


class Param(object):
    def __init__(self, name, min_val, max_val,
                 init_val=None, dist='uniform', scale='log', log_base=np.e, interval=None):
        """

        :param name:
        :param min_val:
        :param max_val:
        :param init_val:
        :param dist:
        :param scale:
        :param log_base:
        :param interval:
        """
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.init_val = init_val
        self.dist = dist
        self.scale = scale
        self.log_base = log_base
        self.interval = interval
        self.param_type = 'continuous'

    def __repr__(self):
        return "%s(%f, %f, %s)" % (self.name, self.min_val, self.max_val, self.scale)

    def get_param_range(self, num, stochastic=False):
        if stochastic:
            if self.dist == 'normal':
                # if Gaussian, then min_val = mean, max_val = sigma
                values = np.random.normal(self.min_val, self.max_val, num)
            else:
                values = np.random.rand(num) * (self.max_val - self.min_val) + self.min_val

            if self.scale == 'log':
                values = np.array([self.log_base ** value for value in values])
        else:
            if self.scale == 'log':
                values = np.logspace(self.min_val, self.max_val, num, base=self.log_base)
            else:
                values = np.linspace(self.min_val, self.max_val, num)

        if self.interval:
            return (np.floor(values / self.interval) * self.interval).astype(int)

        return values

    def get_min(self):
        return self.min_val

    def get_max(self):
        return self.max_val

    def get_type(self):
        if self.interval:
            return 'integer'
        return 'continuous'

    def get_name(self):
        return self.name

    def get_dist(self):
        return self.dist

    def get_scale(self):
        return self.scale
