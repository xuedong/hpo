import os
import numpy as np

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from collections import OrderedDict

import utils
from models import Model
from params import Param

d_ada = OrderedDict()
d_ada['n_estimators'] = ('int', (5, 200))
d_ada['learning_rate'] = ('cont', (-5, -1))


class Ada(Model):
    def __init__(self, problem='binary', n_estimators=50, learning_rate=1):
        self.problem = problem
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.name = 'Ada'

    def eval(self):
        if self.problem == 'binary':
            mod = AdaBoostClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                     random_state=20)
        else:
            mod = AdaBoostRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
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
            arm = {'dir': path + "/" + dirname, 'n_estimators': 50, 'learning_rate': 1., 'results': []}
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
            hps = ['n_estimators', 'learning_rate']
            for hp in hps:
                val = params[hp].get_param_range(1, stochastic=True)
                arm[hp] = val[0]
            arm['results'] = []
            arms[i] = arm

        os.chdir('../../../source')

        return arms

    @staticmethod
    def run_solver(iterations, arm, data,
                   rng=None, problem='cont', method='5fold',
                   track_valid=np.array([1.]), track_test=np.array([1.]), verbose=False):
        """

        :param iterations:
        :param arm:
        :param data:
        :param rng:
        :param problem:
        :param method:
        :param track_valid:
        :param track_test:
        :param verbose:
        :return:
        """
        x, y = data
        loss = utils.Loss(Ada(), x, y, method=method, problem=problem)

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
            current_loss, test_error = loss.evaluate_loss(n_estimators=arm['n_estimators'],
                                                          learning_rate=arm['learning_rate'])
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
            'n_estimators': Param('n_estimators', 5, 200, dist='uniform', scale='linear', interval=1),
            'learning_rate': Param('learning_rate', 1 * 10 ** (-5), 1, dist='uniform', scale='linear')
        }

        return params
