import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import OrderedDict

import utils
from models import Model
from params import Param

d_rf = OrderedDict()
d_rf['n_estimators'] = ('int', (10, 50))
d_rf['min_samples_split'] = ('cont', (0.1, 0.5))
d_rf['max_features'] = ('cont', (0.1, 0.5))


class RF(Model):
    def __init__(self, problem='binary', n_estimators=10, max_features=0.5,
                 min_samples_split=0.3, min_samples_leaf=0.2):
        self.problem = problem
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.name = 'RF'

    def eval(self):
        if self.problem == 'binary':
            mod = RandomForestClassifier(n_estimators=self.n_estimators,
                                         max_features=self.max_features,
                                         min_samples_split=self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf,
                                         n_jobs=-1,
                                         random_state=20)
        else:
            mod = RandomForestRegressor(n_estimators=self.n_estimators,
                                        max_features=self.max_features,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        n_jobs=-1,
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
            arm = {'dir': path + "/" + dirname, 'n_estimators': 10,
                   'min_samples_split': 0.3, 'max_features': 0.5, 'results': []}
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
            hps = ['n_estimators', 'min_samples_split', 'max_features']
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
        loss = utils.Loss(RF(), x, y, method=method, problem=problem)

        best_loss = 1.
        avg_loss = 0.
        test_score = 1.

        if track.size == 0:
            current_best = test_score
            current_track = np.array([1.])
        else:
            current_best = np.amin(track)
            current_track = np.copy(track)

        for iteration in range(iterations):
            current_loss, test_error = loss.evaluate_loss(n_estimators=arm['n_estimators'],
                                                          min_samples_split=arm['min_samples_split'],
                                                          max_features=arm['max_features'])
            current_loss = -current_loss
            avg_loss += current_loss
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

        avg_loss = avg_loss / iterations

        return best_loss, avg_loss, current_track

    @staticmethod
    def get_search_space():
        params = {
            'n_estimators': Param('n_estimators', 10, 50, dist='uniform', scale='linear', interval=1),
            'min_samples_split': Param('min_samples_split', 0.1, 0.5, dist='uniform', scale='linear'),
            'max_features': Param('max_features', 0.1, 0.5, dist='uniform', scale='linear')
        }

        return params
