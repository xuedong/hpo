#!/usr/bin/env python

# This code is based on the original code of Jean-Bastien Grill
# https://team.inria.fr/sequel/Software/POO/

import numpy as np
import math
import os
# import theano.tensor as ts
# import random
# import operator as op
# import pylab as pl
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from six.moves import cPickle
from hyperopt import STATUS_OK

import source.utils as utils
import source.classifiers.logistic as logistic
import source.classifiers.mlp as mlp

# from sklearn.metrics import log_loss, mean_squared_error
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.preprocessing import StandardScaler


# Black box functions

class Sine1(object):
    def __init__(self):
        self.fmax = 0.

    @staticmethod
    def f(x):
        return np.sin(x[0]) - 1.

    def fmax(self):
        return self.fmax


class Sine2(object):
    def __init__(self):
        self.xmax = 3.614
        self.fmax = self.f([self.xmax])

    @staticmethod
    def f(x):
        return -np.cos(x[0])-np.sin(3*x[0])

    def fmax(self):
        return self.fmax


class DoubleSine(object):
    def __init__(self, rho1, rho2, tmax):
        self.ep1 = -math.log(rho1, 2)
        self.ep2 = -math.log(rho2, 2)
        self.tmax = tmax
        self.fmax = 0.

    def f(self, x):
        u = 2*math.fabs(x[0]-self.tmax)

        if u == 0:
            return u
        else:
            ew = math.pow(u, self.ep2) - math.pow(u, self.ep1)
            my_sin = (math.sin(math.pi*math.log(u, 2))+1)/2.
            return my_sin*ew - math.pow(u, self.ep2)

    def fmax(self):
        return self.fmax


class DiffFunc(object):
    def __init__(self, tmax):
        self.tmax = tmax
        self.fmax = 0.

    def f(self, x):
        u = math.fabs(x[0]-self.tmax)
        if u == 0:
            v = 0
        else:
            v = math.log(u, 2)
        fraction = v - math.floor(v)

        if u == 0:
            return u
        elif fraction <= 0.5:
            return -math.pow(u, 2)
        else:
            return -math.sqrt(u)

    def fmax(self):
        return self.fmax


class Garland(object):
    def __init__(self):
        self.fmax = 0.997772313413222

    @staticmethod
    def f(x):
        return 4*x[0]*(1-x[0])*(0.75+0.25*(1-np.sqrt(np.abs(np.sin(60*x[0])))))

    def fmax(self):
        return self.fmax


class Himmelblau(object):
    def __init__(self):
        self.fmax = 0.

    @staticmethod
    def f(x):
        return -(x[0]**2+x[1]-11.)**2-(x[0]+x[1]**2-7.)**2

    def fmax(self):
        return self.fmax


class Rosenbrock(object):
    def __init__(self, a, b):
        self.fmax = 0.
        self.a = a
        self.b = b

    def f(self, x):
        return -(self.a-x[0])**2-self.b*(x[1]-x[0]**2)**2

    def fmax(self):
        return self.fmax


class Gramacy1(object):
    def __init__(self):
        self.xmax = 0.54856343
        self.fmax = self.f([self.xmax])

    @staticmethod
    def f(x):
        return -np.sin(10*np.pi*x[0])/(2*x[0])-(x[0]-1)**4

    def fmax(self):
        return self.fmax


# Scikit-learn functions

class SklearnSVM(object):
    def __init__(self, model, x, y, method, problem, director):
        self.director = director
        self.loss = utils.Loss(model, x, y, method, problem)

    def f(self, x):
        valid_error, test_error = self.loss.evaluate_loss(c=10**x[0], gamma=10**x[1])
        with open(self.director + '/tracks.pkl', 'wb') as file:
            cPickle.dump([valid_error, test_error], file)
        return valid_error


class SklearnGBM(object):
    def __init__(self, model, x, y, method, problem, director):
        self.director = director
        self.loss = utils.Loss(model, x, y, method, problem)

    def f(self, x):
        valid_error, test_error = self.loss.evaluate_loss(learning_rate=10**x[0], n_estimators=x[1],
                                                          max_depth=x[2], min_samples_split=x[3])
        with open(self.director + '/tracks.pkl', 'wb') as file:
            cPickle.dump([valid_error, test_error], file)
        return valid_error


class SklearnKNN(object):
    def __init__(self, model, x, y, method, problem, director):
        self.director = director
        self.loss = utils.Loss(model, x, y, method, problem)

    def f(self, x):
        valid_error, test_error = self.loss.evaluate_loss(n_neighbors=x[0])
        with open(self.director + '/tracks.pkl', 'wb') as file:
            cPickle.dump([valid_error, test_error], file)
        return valid_error


class SklearnMLP(object):
    def __init__(self, model, x, y, method, problem, director):
        self.director = director
        self.loss = utils.Loss(model, x, y, method, problem)

    def f(self, x):
        valid_error, test_error = self.loss.evaluate_loss(hidden_layer_size=x[0], alpha=x[1])
        with open(self.director + '/tracks.pkl', 'wb') as file:
            cPickle.dump([valid_error, test_error], file)
        return valid_error


class SklearnRF(object):
    def __init__(self, model, x, y, method, problem, director):
        self.director = director
        self.loss = utils.Loss(model, x, y, method, problem)

    def f(self, x):
        valid_error, test_error = self.loss.evaluate_loss(n_estimators=x[0], min_samples_split=x[1], max_features=x[2])
        with open(self.director + '/tracks.pkl', 'wb') as file:
            cPickle.dump([valid_error, test_error], file)
        return valid_error


class SklearnTree(object):
    def __init__(self, model, x, y, method, problem, director):
        self.director = director
        self.loss = utils.Loss(model, x, y, method, problem)

    def f(self, x):
        valid_error, test_error = self.loss.evaluate_loss(max_features=x[0], max_depth=x[1], min_samples_split=x[2])
        with open(self.director + '/tracks.pkl', 'wb') as file:
            cPickle.dump([valid_error, test_error], file)
        return valid_error


class SklearnAda(object):
    def __init__(self, model, x, y, method, problem, director):
        self.director = director
        self.loss = utils.Loss(model, x, y, method, problem)

    def f(self, x):
        valid_error, test_error = self.loss.evaluate_loss(n_estimators=x[0], learning_rate=10**x[1])
        with open(self.director + '/tracks.pkl', 'wb') as file:
            cPickle.dump([valid_error, test_error], file)
        return valid_error


# Theano functions

class TheanoHCTLogistic(object):
    def __init__(self, epochs, data, director):
        self.epochs = epochs
        self.data = data
        self.director = director
        self.change_status = False

    def f(self, hps):
        arm = {'dir': self.director, 'learning_rate': np.exp(hps[0]), 'batch_size': int(hps[1]), 'results': []}
        if not os.path.exists(self.director + '/best_model.pkl') or self.change_status:
            train_loss, best_valid_loss, test_score, track_valid, track_test = \
                logistic.LogisticRegression.run_solver(self.epochs, arm, self.data, verbose=False)
        else:
            classifier = cPickle.load(open(self.director + '/best_model.pkl', 'rb'))
            train_loss, best_valid_loss, test_score, track_valid, track_test = \
                logistic.LogisticRegression.run_solver(self.epochs, arm, self.data,
                                                       classifier=classifier, verbose=False)

        with open(self.director + '/tracks.pkl', 'wb') as file:
            cPickle.dump([best_valid_loss, test_score], file)

        return best_valid_loss

    def set_status(self, flag):
        self.change_status = flag
        # print(self.change_status)


class TheanoHOOLogistic(object):
    def __init__(self, epochs, data, director):
        self.epochs = epochs
        self.data = data
        self.director = director
        self.change_status = False

    def f(self, hps):
        arm = {'dir': self.director, 'learning_rate': np.exp(hps[0]), 'batch_size': int(hps[1]), 'results': []}

        train_loss, best_valid_loss, test_score, track_valid, track_test = \
            logistic.LogisticRegression.run_solver(self.epochs, arm, self.data, verbose=True)

        with open(self.director + '/tracks.pkl', 'wb') as file:
            cPickle.dump([track_valid, track_test], file)
        # print(track_valid)
        # print(track_test)

        return best_valid_loss


class TheanoHCTMLP(object):
    def __init__(self, epochs, data, director):
        self.epochs = epochs
        self.data = data
        self.director = director
        self.change_status = False

    def f(self, hps):
        arm = {'dir': self.director, 'learning_rate': np.exp(hps[0]),
               'batch_size': int(hps[1]), 'n_hidden': 500, 'l1_reg': 0., 'l2_reg': np.exp(hps[2]),
               'results': []}
        if not os.path.exists(self.director + '/best_model.pkl') or self.change_status:
            train_loss, best_valid_loss, test_score, track_valid, track_test = \
                mlp.MLP.run_solver(self.epochs, arm, self.data, verbose=True)
        else:
            classifier = cPickle.load(open(self.director + '/best_model.pkl', 'rb'))
            train_loss, best_valid_loss, test_score, track_valid, track_test = \
                mlp.MLP.run_solver(self.epochs, arm, self.data, classifier=classifier, verbose=True)

        with open(self.director + '/tracks.pkl', 'wb') as file:
            cPickle.dump([best_valid_loss, test_score], file)

        return best_valid_loss

    def set_status(self, flag):
        self.change_status = flag
        # print(self.change_status)


class TheanoHOOMLP(object):
    def __init__(self, epochs, data, director):
        self.epochs = epochs
        self.data = data
        self.director = director
        self.change_status = False

    def f(self, hps):
        arm = {'dir': self.director, 'learning_rate': np.exp(hps[0]),
               'batch_size': int(hps[1]), 'n_hidden': 500, 'l1_reg': 0., 'l2_reg': np.exp(hps[2]),
               'results': []}

        train_loss, best_valid_loss, test_score, track_valid, track_test = \
            mlp.MLP.run_solver(self.epochs, arm, self.data, verbose=True)

        with open(self.director + '/tracks.pkl', 'wb') as file:
            cPickle.dump([track_valid, track_test], file)
        # print(track_valid)
        # print(track_test)

        return best_valid_loss


# Hyperopt functions

class HyperLogistic(object):
    def __init__(self, model, epochs, director, data):
        self.model = model
        self.epochs = epochs
        self.director = director
        self.data = data

    def objective(self, hps):
        learning_rate, batch_size = hps
        arm = {'dir': self.director,
               'learning_rate': learning_rate, 'batch_size': int(batch_size),
               'results': []}
        train_loss, best_valid_loss, test_score, track_valid, track_test = \
            self.model.run_solver(self.epochs, arm, self.data, verbose=False)
        return {
            'loss': best_valid_loss,
            'status': STATUS_OK,
            # -- store other results like this
            'train_loss': train_loss,
            'valid_loss': best_valid_loss,
            # -- attachments are handled differently
            'attachments':
                {'track_valid': cPickle.dumps(track_valid),
                 'track_test': cPickle.dumps(track_test)}
        }


class HyperMLP(object):
    def __init__(self, model, epochs, director, data):
        self.model = model
        self.epochs = epochs
        self.director = director
        self.data = data

    def objective(self, hps):
        learning_rate, batch_size, l2_reg = hps
        arm = {'dir': self.director,
               'learning_rate': learning_rate, 'batch_size': int(batch_size),
               'n_hidden': 500, 'l1_reg': 0., 'l2_reg': l2_reg,
               'results': []}
        train_loss, best_valid_loss, test_score, track_valid, track_test = \
            self.model.run_solver(self.epochs, arm, self.data, verbose=True)
        return {
            'loss': best_valid_loss,
            'status': STATUS_OK,
            # -- store other results like this
            'train_loss': train_loss,
            'valid_loss': best_valid_loss,
            # -- attachments are handled differently
            'attachments':
                {'track_valid': cPickle.dumps(track_valid),
                 'track_test': cPickle.dumps(track_test)}
        }


class HyperAda(object):
    def __init__(self, model, iterations, director, data):
        self.model = model
        self.iterations = iterations
        self.director = director
        self.data = data

    def objective(self, hps):
        n_estimators, learning_rate = hps
        arm = {'dir': self.director,
               'n_estimators': n_estimators, 'learning_rate': learning_rate,
               'results': []}
        best_loss, avg_loss, track_valid, track_test = \
            self.model.run_solver(self.iterations, arm, self.data, verbose=True)
        return {
            'loss': best_loss,
            'status': STATUS_OK,
            # -- store other results like this
            'average_loss': avg_loss,
            # -- attachments are handled differently
            'attachments':
                {'track_valid': cPickle.dumps(track_valid),
                 'track_test': cPickle.dumps(track_test)}
        }
