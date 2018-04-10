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
        loss = utils.Loss(Ada(), x, y, method=method, problem=problem)

        best_loss = 1.
        test_score = 1.

        if track.size == 0:
            current_best = test_score
            current_track = np.array([1.])
        else:
            current_best = np.amin(track)
            current_track = np.copy(track)

        for iteration in range(iterations):
            current_loss = -loss.evaluate_loss(n_estimators=arm['n_estimators'], learning_rate=arm['learning_rate'])

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
            'n_estimators': Param('n_estimators', 5, 200, dist='uniform', scale='linear', interval=1),
            'learning_rate': Param('learning_rate', 1 * 10 ** (-5), 1, dist='uniform', scale='linear')
        }

        return params
