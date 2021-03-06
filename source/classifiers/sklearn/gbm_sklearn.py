import os
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from collections import OrderedDict

import utils
from models import Model
from params import Param

d_gbm = OrderedDict()
d_gbm['learning_rate'] = ('cont', (-5, -1))
d_gbm['n_estimators'] = ('int', (10, 100))
d_gbm['max_depth'] = ('int', (2, 100))
d_gbm['min_samples_split'] = ('int', (2, 100))


class GBM(Model):
    def __init__(self, problem='binary', learning_rate=0.1, n_estimators=100, max_depth=3, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, subsample=1.0, max_features=1.0):
        self.problem = problem
        self.learning_rate = learning_rate
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.name = 'GBM'

    def eval(self):
        if self.problem == 'binary':
            mod = GradientBoostingClassifier(learning_rate=self.learning_rate, n_estimators=self.n_estimators,
                                             max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                             min_samples_leaf=self.min_samples_leaf,
                                             min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                             subsample=self.subsample,
                                             max_features=self.max_features,
                                             random_state=20)
        else:
            mod = GradientBoostingRegressor(learning_rate=self.learning_rate, n_estimators=self.n_estimators,
                                            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                            min_samples_leaf=self.min_samples_leaf,
                                            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                            subsample=self.subsample,
                                            max_features=self.max_features,
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
            arm = {'dir': path + "/" + dirname, 'learning_rate': 0.1, 'n_estimators': 200,
                   'max_depth': 3, 'min_samples_split': 2, 'results': []}
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
            hps = ['learning_rate', 'n_estimators', 'max_depth', 'min_samples_split']
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
        loss = utils.Loss(GBM(), x, y, x_test, y_test, method=method, problem=problem)

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
            current_loss, test_error = loss.evaluate_loss(learning_rate=arm['learning_rate'],
                                                          n_estimators=arm['n_estimators'],
                                                          max_depth=arm['max_depth'],
                                                          min_samples_split=arm['min_samples_split'])
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
            'learning_rate': Param('learning_rate', 1 * 10 ** (-5), 1 * 10 ** (-1), dist='uniform', scale='linear'),
            'n_estimators': Param('n_estimators', 10, 100, dist='uniform', scale='linear', interval=1),
            'max_depth': Param('max_depth', 2, 100, dist='uniform', scale='linear', interval=1),
            'min_samples_split': Param('min_samples_split', 2, 100, dist='uniform', scale='linear', interval=1)
        }

        return params
