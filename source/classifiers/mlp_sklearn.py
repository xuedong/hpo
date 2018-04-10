import os
import numpy as np

from sklearn.neural_network import MLPClassifier, MLPRegressor
from collections import OrderedDict

import utils
from models import Model
from params import Param

d_mlp = OrderedDict()
d_mlp['hidden_layer_size'] = ('int', (5, 50))
d_mlp['alpha'] = ('cont', (1e-5, 0.9))


class MLP(Model):
    def __init__(self, problem='binary', hidden_layer_size=100, alpha=10e-4,
                 learning_rate_init=10e-4, beta_1=0.9, beta_2=0.999):
        self.problem = problem
        self.hidden_layer_sizes = (int(hidden_layer_size),)
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.name = 'MLP'

    def eval(self):
        if self.problem == 'binary':
            mod = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                                learning_rate_init=self.learning_rate_init, beta_1=self.beta_1, beta_2=self.beta_2,
                                random_state=20)
        else:
            mod = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
                               learning_rate_init=self.learning_rate_init, beta_1=self.beta_1, beta_2=self.beta_2,
                               random_state=20)
        return mod

    @staticmethod
    def generate_arms(n, path, params, default=False):
        """Function that generates a dictionary of configurations/arms.

        :param n: number of arms to generate
        :param path: path to which we store the results later
        :param params: hyperparameter to be optimized
        :param default: default arm option
        :return:
        """
        os.chdir(path)
        arms = {}
        if default:
            dirname = "default_arm"
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            arm = {'dir': path + "/" + dirname, 'hidden_layer_size': 100, 'alpha': 10 ** (-4), 'results': []}
            arms[0] = arm
            return arms
        subdirs = next(os.walk('.'))[1]
        if len(subdirs) == 0:
            start_count = 0
        else:
            start_count = len(subdirs)
        for i in range(n):
            dirname = "arm" + str(start_count + i)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            arm = {'dir': path + "/" + dirname}
            hps = ['hidden_layer_size', 'alpha']
            for hp in hps:
                val = params[hp].get_param_range(1, stochastic=True)
                arm[hp] = val[0]
            arm['results'] = []
            arms[i] = arm

        os.chdir('../../../source')

        return arms

    @staticmethod
    def run_solver(iterations, arm, data,
                   rng=None, problem='cont', method='5fold', track=np.array([1.]), verbose=False):
        """

        :param iterations:
        :param arm:
        :param data:
        :param rng:
        :param problem:
        :param method:
        :param track:
        :param verbose:
        :return:
        """
        x, y = data
        loss = utils.Loss(MLP(), x, y, method=method, problem=problem)

        best_loss = 1.
        test_score = 1.

        if track.size == 0:
            current_best = test_score
            current_track = np.array([1.])
        else:
            current_best = np.amin(track)
            current_track = np.copy(track)

        for iteration in range(iterations):
            current_loss, test_error = loss.evaluate_loss(hidden_layer_size=arm['hidden_layer_size'],
                                                          alpha=arm['alpha'])
            current_loss = -current_loss
            test_score = -test_error

            if verbose:
                print(
                    'iteration %i, validation error %f %%' %
                    (
                        iteration,
                        current_loss * 100.
                    )
                )

            if current_loss < best_loss:
                best_loss = current_loss
                # best_iter = iteration

            if test_score < current_best:
                current_track = np.append(current_track, test_score)
            else:
                current_track = np.append(current_track, current_best)
        return best_loss, current_track

    @staticmethod
    def get_search_space():
        params = {
            'hidden_layer_size': Param('hidden_layer_size', 5, 50, dist='uniform', scale='linear', interval=1),
            'alpha': Param('alpha', 1 * 10 ** (-5), 0.9, dist='uniform', scale='linear')
        }

        return params
