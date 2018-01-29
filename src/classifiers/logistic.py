from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self, input, n, m):
        """Initialization of the class.

        :param input: one minibatch
        :param n: dimension of the input space
        :param m: dimension of the output space
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
        # parameters of the model
        self.params = [self.W, self.b]
        # keep track of the input
        self.input = input

    def neg_log_likelihood(self, y):
        """Log-likelihood loss.

        :param y: correct label vector
        :return: the mean of the negative log-likelihood of the prediction, we use mean instead of sum here
        to make the learning rate less dependent of the size of the minibatch size
        """
        return -T.mean(T.log(self.p_y_x)[T.arange(y.shape[0]), y])

    def zero_one(self, y):
        """Zero-one loss.

        :param y: correct label vector
        :return: the zero-one loss over the size of minibatch
        """
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
