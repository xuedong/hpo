from __future__ import print_function

import sys
import timeit
import os
import numpy as np

from six.moves import cPickle

import theano
import theano.tensor as ts

from models import Model
from params import Param
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


class MLP(Model):
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

    def generate_arms(self, n, path, params, default=False):
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
            arm = {'dir': path + "/" + dirname,
                   'learning_rate': 0.001, 'batch_size': 100, 'n_hidden': 500,
                   'l1_reg': 1., 'l2_reg': 0., 'results': []}
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
            hps = ['learning_rate', 'batch_size', 'l1_reg']
            for hp in hps:
                val = params[hp].get_param_range(1, stochastic=True)
                arm[hp] = val[0]
            arm['l2_reg'] = 0.
            arm['n_hidden'] = 500
            arm['results'] = []
            arms[i] = arm

        os.chdir('../../../src')

        return arms


def run_solver(epochs, arm, data, rng, classifier=None, track=np.array([1.]), verbose=False):
    """

    :param epochs:
    :param arm:
    :param data:
    :param rng:
    :param classifier:
    :param track:
    :param verbose:
    :return:
    """
    train_input, train_target = data[0]
    valid_input, valid_target = data[1]
    test_input, test_target = data[2]

    n_batches_train = train_input.get_value(borrow=True).shape[0] // arm['batch_size']
    n_batches_valid = valid_input.get_value(borrow=True).shape[0] // arm['batch_size']
    n_batches_test = test_input.get_value(borrow=True).shape[0] // arm['batch_size']

    if verbose:
        print('Building model...')

    # symbolic variables
    index = ts.lscalar()
    x = ts.matrix('x')
    y = ts.ivector('y')

    # construct the classifier
    if not classifier:
        classifier = MLP(rng=rng, input_data=x, n_in=28*28, n_hidden=arm['n_hidden'], n_out=10)
    cost = classifier.neg_log_likelihood(y) + arm['l1_reg'] * classifier.l1 + arm['l2_reg'] * classifier.l2

    # construct a Theano function that computes the errors made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.zero_one(y),
        givens={
            x: test_input[index * arm['batch_size']: (index + 1) * arm['batch_size']],
            y: test_target[index * arm['batch_size']: (index + 1) * arm['batch_size']]
        }
    )
    valid_model = theano.function(
        inputs=[index],
        outputs=classifier.zero_one(y),
        givens={
            x: valid_input[index * arm['batch_size']: (index + 1) * arm['batch_size']],
            y: valid_target[index * arm['batch_size']: (index + 1) * arm['batch_size']]
        }
    )

    # construct a Theano function that updates the parameters of
    # the training model using stochastic gradient descent
    g_params = [ts.grad(cost=cost, wrt=param) for param in classifier.params]
    updates = [
        (param, param - arm['learning_rate'] * g_param)
        for param, g_param in zip(classifier.params, g_params)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_input[index * arm['batch_size']: (index + 1) * arm['batch_size']],
            y: train_target[index * arm['batch_size']: (index + 1) * arm['batch_size']]
        }
    )

    if verbose:
        print('Training model...')

    # early-stopping parameters
    patience = 10000
    patience_increase = 2
    threshold = 0.995
    valid_freq = min(n_batches_train, patience // 2)

    best_valid_loss = np.inf
    best_iter = 0
    test_score = 0.
    train_loss = 0.

    if track.size == 0:
        current_best = 1.
        current_track = np.array([1.])
    else:
        current_best = np.amin(track)
        current_track = np.copy(track)

    start_time = timeit.default_timer()
    done = False
    epoch = 0
    while (epoch < epochs) and not done:
        epoch += 1

        if best_valid_loss < current_best:
            current_track = np.append(current_track, test_score)
        else:
            current_track = np.append(current_track, current_best)

        for batch_index in range(n_batches_train):
            batch_cost = train_model(batch_index)
            iteration = (epoch - 1) * n_batches_train + batch_index

            if (iteration + 1) % valid_freq == 0:
                valid_losses = [valid_model(i) for i in range(n_batches_valid)]
                current_valid_loss = float(np.mean(valid_losses))

                if verbose:
                    print(
                        'epoch %i, batch %i/%i, batch average cost %f, validation error %f %%' %
                        (
                            epoch,
                            batch_index + 1,
                            n_batches_train,
                            batch_cost,
                            current_valid_loss * 100.
                        )
                    )

                if current_valid_loss < best_valid_loss:
                    if current_valid_loss < best_valid_loss * threshold:
                        patience = max(patience, iteration * patience_increase)

                    best_valid_loss = current_valid_loss
                    best_iter = iteration

                    train_losses = [train_model(i) for i in range(n_batches_train)]
                    train_loss = np.mean(train_losses)

                    test_losses = [test_model(i) for i in range(n_batches_test)]
                    test_score = np.mean(test_losses)

                    if verbose:
                        print(
                            (
                                '     epoch %i, batch %i/%i, test error of'
                                ' best model %f %%'
                            ) %
                            (
                                epoch,
                                batch_index + 1,
                                n_batches_train,
                                test_score * 100.
                            )
                        )

                    # save the best model
                    with open('../log/best_model_mlp_sgd.pkl', 'wb') as file:
                        cPickle.dump(classifier, file)

            if patience <= iteration:
                done = True
                break

    end_time = timeit.default_timer()

    if verbose:
        print(
            (
                'Optimization completed with best validation score of %f %%, '
                'obtained at iteration %i, with test performance %f %%'
            )
            % (best_valid_loss * 100., best_iter + 1, test_score * 100.)
        )
        print('The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time)))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    return train_loss, best_valid_loss, test_score, current_track


def get_search_space():
    params = {
        'learning_rate': Param('learning_rate', np.log(1 * 10 ** (-3)), np.log(1 * 10 ** (-1)), dist='uniform',
                               scale='log'),
        'batch_size': Param('batch_size', 1, 1000, dist='uniform', scale='linear', interval=1),
        'l1_reg': Param('l1_reg', np.log(1 * 10 ** (-3)), np.log(1), dist='uniform', scale='log')
    }

    return params
