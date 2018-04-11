import os
import numpy as np

from sklearn.svm import SVC, SVR
from collections import OrderedDict

import utils
from models import Model
from params import Param

d_svm = OrderedDict()
d_svm['c'] = ('cont', (-4, 5))
d_svm['gamma'] = ('cont', (-4, 5))


class SVM(Model):
    def __init__(self, problem='binary', c=0, gamma=0, kernel='rbf'):
        self.problem = problem
        self.c = c
        self.gamma = gamma
        self.kernel = kernel
        self.name = 'SVM'

    def eval(self):
        if self.problem == 'binary':
            mod = SVC(kernel=self.kernel, C=self.c, gamma=self.gamma, probability=True, random_state=20)
        else:
            mod = SVR(kernel=self.kernel, C=self.c, gamma=self.gamma)
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
            arm = {'dir': path + "/" + dirname, 'c': 0., 'gamma': 1., 'results': []}
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
            hps = ['c', 'gamma']
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
        loss = utils.Loss(SVM(), x, y, method=method, problem=problem)

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
            current_loss, test_error = loss.evaluate_loss(c=arm['c'], gamma=arm['gamma'])
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
            'c': Param('c', np.log(1 * 10 ** (-5)), np.log(1 * 10 ** 5), dist='uniform', scale='log'),
            'gamma': Param('gamma', np.log(1 * 10 ** (-5)), np.log(1 * 10 ** 5), dist='uniform', scale='log')
        }

        return params
