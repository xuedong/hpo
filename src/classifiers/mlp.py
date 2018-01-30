from __future__ import print_function

import numpy as np

import theano
import theano.tensor as ts

from classifiers.logistic import LogisticRegression


class HiddenLayer(object):
    def __init__(self, rng, input_data, n, m, w=None, b=None, activation=ts.tanh):
        """Hidden layer of a MLP.

        :param rng: random state, used to initialize weights
        :param input_data: input tensor of shape (n_examples, n)
        :param n: dimension of input
        :param m: number of hidden units
        :param w: weight matrix
        :param b: bias matrix
        :param activation: activation function, set to tanh by default
        """
        self.input_data = input_data
        # initialize self.w by uniformly sampling from some interval
        # defined in [Xavier10] (4x times larger if sigmoid activation function
        if w is None:
            w_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n + m)),
                    high=np.sqrt(6. / (n + m)),
                    size=(n, m)
                ),
                dtype=theano.config.floatX
            )
            if activation == ts.nnet.sigmoid:
                w_values *= 4.
            w = theano.shared(value=w_values, name='w', borrow=True)
        self.w = w
        # initialize self.b with zeros
        if b is None:
            b_values = np.zeros((m,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.b = b
        # output of this hidden layer
        output = ts.dot(input_data, self.w) + self.b
        self.output = output if activation is None else activation(output)
        # parameters of the model
        self.params = [self.w, self.b]


class MLP(object):
    def __init__(self, rng, input_data, n_in, n_hidden, n_out):
        """Single layer mlp.

        :param rng: random state
        :param input_data: typically a minibatch of input
        :param n_in: number of input units
        :param n_hidden: number of hidden units
        :param n_out: number of output units
        """
        self.hidden_layer = HiddenLayer(rng=rng, input_data=input_data, n=n_in, m=n_hidden)
        self.logistic_layer = LogisticRegression(input_data=self.hidden_layer.output, n=n_hidden, m=n_out)
        # regularization
        self.l1 = abs(self.hidden_layer.w).sum() + abs(self.logistic_layer.w).sum()
        self.l2 = (self.hidden_layer.w ** 2).sum() + (self.logistic_layer.w ** 2).sum()
        # loss functions
        self.neg_log_likelihood = self.logistic_layer.neg_log_likelihood
        self.zero_one = self.logistic_layer.zero_one
        # parameters of the model
        self.params = self.hidden_layer.params + self.logistic_layer.params
        # keep track of the input
        self.input_data = input_data
