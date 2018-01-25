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
        """

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