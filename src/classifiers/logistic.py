from __future__ import print_function

import six.moves.cPickle
import gzip
import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self, input, n, m):
        """Initialization of the class.

        :param input: one minibatch
        :param n: the dimension of the input space
        :param m: the dimension of the output space
        """
        # initialize the weight matrix W
        self.W = theano.shared(
            value=np.zeros(
                (n, m),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the bias vector b
        self.b = theano.shared(
            value=np.zeros(
                (n, 1),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        # compute the matrix of class-membership probabilities
        self.p_y_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # predict the class
        self.y_pred = T.argmax(self.p_y_x, axis=1)

    def neg_log_likelihood(self, y):
        """Loss function.

        :param y: the correct label vector
        :return: the mean of the negative log-likelihood of the prediction, we use mean instead of sum here
        to make the learning rate less dependent of the size of the minibatch size
        """
        return -T.mean(T.log(self.p_y_x)[T.arange(y.shape[0]), y])