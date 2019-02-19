import os
import numpy as np

from sklearn.neural_network import MLPClassifier, MLPRegressor
from collections import OrderedDict

import utils
from models import Model
from params import Param

d_mlp = OrderedDict()
d_mlp['hidden_layer_size'] = ('int', (5, 100))
d_mlp['alpha'] = ('cont', (1e-5, 0.9))
d_mlp['learning_rate_init'] = ('cont', (-5, -1))


class MLP(Model):
    def __init__(self, problem='binary', hidden_layer_size=100, alpha=10e-4,
                 learning_rate_init=1e-4, beta_1=0.9, beta_2=0.999):
        self.problem = problem
        self.hidden_layer_sizes = (int(hidden_layer_size),)
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.name = 'MLP'

    def eval(self):
        if self.problem == 'binary':
            mod = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=10, alpha=self.alpha, solver='sgd',
                                learning_rate_init=self.learning_rate_init, early_stopping=True,
                                beta_1=self.beta_1, beta_2=self.beta_2, random_state=20)
        else:
            mod = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=10, alpha=self.alpha, solver='sgd',
                               learning_rate_init=self.learning_rate_init, early_stopping=True,
                               beta_1=self.beta_1, beta_2=self.beta_2, random_state=20)
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
            arm = {'dir': path + "/" + dirname, 'hidden_layer_size': 100, 'alpha': 10 ** (-4),
                   'learning_rate_init': 0.001, 'results': []}
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
            hps = ['hidden_layer_size', 'alpha', 'learning_rate_init']
            for hp in hps:
                val = params[hp].get_param_range(1, stochastic=True)
                arm[hp] = val[0]
            arm['results'] = []
            arms[i] = arm

        os.chdir('../../../source')

        return arms

    @staticmethod
    def run_solver(iterations, arm, data, test,
                   rng=None, problem='cont', method='5fold',
                   track_valid=np.array([1.]), track_test=np.array([1.]), verbose=False):
        """

        :param iterations:
        :param arm:
        :param data:
        :param test:
        :param rng:
        :param problem:
        :param method:
        :param track_valid:
        :param track_test:
        :param verbose:
        :return:
        """
        x, y = data
        x_test, y_test = test
        loss = utils.Loss(MLP(), x, y, x_test, y_test, method=method, problem=problem)

        best_loss = 1.
        avg_loss = 0.
        test_score = 1.

        if track_valid.size == 0:
            current_best_valid = 1.
            current_test = 1.
            current_track_valid = np.array([1.])
            current_track_test = np.array([1.])
        else:
            current_best_valid = track_valid[-1]
            current_test = track_test[-1]
            current_track_valid = np.copy(track_valid)
            current_track_test = np.copy(track_test)

        for iteration in range(iterations):
            current_loss, test_error = loss.evaluate_loss(hidden_layer_size=arm['hidden_layer_size'],
                                                          alpha=arm['alpha'],
                                                          learning_rate_init=arm['learning_rate_init'])
            current_loss = -current_loss
            avg_loss += current_loss

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
                test_score = -test_error
                # best_iter = iteration

            if best_loss < current_best_valid:
                current_best_valid = best_loss
                current_test = test_score
                current_track_valid = np.append(current_track_valid, current_best_valid)
                current_track_test = np.append(current_track_test, current_test)
            else:
                current_track_valid = np.append(current_track_valid, current_best_valid)
                current_track_test = np.append(current_track_test, current_test)

        avg_loss = avg_loss / iterations

        return best_loss, avg_loss, current_track_valid, current_track_test

    @staticmethod
    def get_search_space():
        params = {
            'hidden_layer_size': Param('hidden_layer_size', 5, 50, dist='uniform', scale='linear', interval=1),
            'alpha': Param('alpha', 1 * 10 ** (-5), 0.9, dist='uniform', scale='linear'),
            'learning_rate_init': Param('learning_rate_init', np.log(1 * 10 ** (-5)), np.log(1 * 10 ** (-1)),
                                        dist='uniform', scale='log')
        }

        return params
