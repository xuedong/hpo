#!/usr/bin/env python

# This code is based on the original code of Jean-Bastien Grill
# https://team.inria.fr/sequel/Software/POO/

import numpy as np
import math
import theano.tensor as ts
# import random
# import operator as op
# import pylab as pl
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import src.utils as utils
import src.classifiers.logistic as logistic

# from sklearn.metrics import log_loss, mean_squared_error
# from sklearn.model_selection import train_test_split, KFold
# from sklearn.preprocessing import StandardScaler


# Black box functions

class Sine1:
    def __init__(self):
        self.fmax = 0.

    @staticmethod
    def f(x):
        return np.sin(x[0]) - 1.

    def fmax(self):
        return self.fmax


class Sine2:
    def __init__(self):
        self.xmax = 3.614
        self.fmax = self.f([self.xmax])

    @staticmethod
    def f(x):
        return -np.cos(x[0])-np.sin(3*x[0])

    def fmax(self):
        return self.fmax


class DoubleSine:
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


class DiffFunc:
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


class Garland:
    def __init__(self):
        self.fmax = 0.997772313413222

    @staticmethod
    def f(x):
        return 4*x[0]*(1-x[0])*(0.75+0.25*(1-np.sqrt(np.abs(np.sin(60*x[0])))))

    def fmax(self):
        return self.fmax


class Himmelblau:
    def __init__(self):
        self.fmax = 0.

    @staticmethod
    def f(x):
        return -(x[0]**2+x[1]-11.)**2-(x[0]+x[1]**2-7.)**2

    def fmax(self):
        return self.fmax


class Rosenbrock:
    def __init__(self, a, b):
        self.fmax = 0.
        self.a = a
        self.b = b

    def f(self, x):
        return -(self.a-x[0])**2-self.b*(x[1]-x[0]**2)**2

    def fmax(self):
        return self.fmax


class Gramacy1:
    def __init__(self):
        self.xmax = 0.54856343
        self.fmax = self.f([self.xmax])

    @staticmethod
    def f(x):
        return -np.sin(10*np.pi*x[0])/(2*x[0])-(x[0]-1)**4

    def fmax(self):
        return self.fmax


# Scikit-learn functions

class SklearnSVM:
    def __init__(self, model, x, y, method, problem):
        self.loss = utils.Loss(model, x, y, method, problem)

    def f(self, x):
        return self.loss.evaluate_loss(C=x[0], gamma=x[1])


class SklearnGBM:
    def __init__(self, model, x, y, method, problem):
        self.loss = utils.Loss(model, x, y, method, problem)

    def f(self, x):
        return self.loss.evaluate_loss(learning_rate=x[0], n_estimators=x[1], max_depth=x[2], min_samples_split=x[3])


class SklearnKNN:
    def __init__(self, model, x, y, method, problem):
        self.loss = utils.Loss(model, x, y, method, problem)

    def f(self, x):
        return self.loss.evaluate_loss(n_neighbors=x[0])


class SklearnMLP:
    def __init__(self, model, x, y, method, problem):
        self.loss = utils.Loss(model, x, y, method, problem)

    def f(self, x):
        return self.loss.evaluate_loss(hidden_layer_size=x[0], alpha=x[1])


# Theano functions

class TheanoLogistic:
    def __init__(self, epochs, data, director):
        x = ts.matrix('x')
        self.model = logistic.LogisticRegression(x, 28*28, 10)
        self.epochs = epochs
        self.data = data
        self.track = None
        self.director = director

    def f(self, x):
        arm = {'dir': self.director, 'learning_rate': np.exp(x[0]), 'batch_size': int(x[1]), 'results': []}
        train_loss, best_valid_loss, test_score, track = self.model.run_solver(self.epochs, arm, self.data)
        self.track = track
        return -test_score