from __future__ import print_function

import numpy as np
import os
import timeit
import sys

from six.moves import cPickle

import theano
import theano.tensor as ts

# import utils
from models import Model
from params import Param


class LogisticRegression(Model):
    def __init__(self, input_data, n, m):
        """Initialization of the class.

        :param input_data: one minibatch
        :param n: dimension of the input space
        :param m: dimension of the output space
        """
        # initialize the weight matrix W
        self.w = theano.shared(
            value=np.zeros(
                (n, m),
                dtype=theano.config.floatX
            ),
            name='w',
            borrow=True
        )
        # initialize the bias vector b
        self.b = theano.shared(
            value=np.zeros(
                (m,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        # compute the matrix of class-membership probabilities
        self.p_y_x = ts.nnet.softmax(ts.dot(input_data, self.w) + self.b)
        # predict the class
        self.y_pred = ts.argmax(self.p_y_x, axis=1)
        # parameters of the model
        self.params = [self.w, self.b]
        # keep track of the input
        self.input_data = input_data

    def neg_log_likelihood(self, y):
        """Log-likelihood loss.

        :param y: correct label vector
        :return: the mean of the negative log-likelihood of the prediction, we use mean instead of sum here
        to make the learning rate less dependent of the size of the minibatch size
        """
        return -ts.mean(ts.log(self.p_y_x)[ts.arange(y.shape[0]), y])

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
            return ts.mean(ts.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

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
            arm = {'dir': path + "/" + dirname, 'learning_rate': 0.001, 'batch_size': 100, 'results': []}
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
            hps = ['learning_rate', 'batch_size']
            for hp in hps:
                val = params[hp].get_param_range(1, stochastic=True)
                arm[hp] = val[0]
            arm['results'] = []
            arms[i] = arm

        os.chdir('../../../src')

        return arms

    @staticmethod
    def run_solver(epochs, arm, data, rng=None, classifier=None, track=np.array([1.]), verbose=False):
        """

        :param epochs: number of epochs
        :param arm: hyperparameter configuration encoded as a dictionary
        :param data: dataset to use
        :param rng: not used here
        :param classifier: initial model, set as None by default
        :param track: vector where we store the test errors
        :param verbose: verbose option
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

        # construct a classifier
        if not classifier:
            classifier = LogisticRegression(input_data=x, n=28*28, m=10)
        cost = classifier.neg_log_likelihood(y)

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
        g_w = ts.grad(cost=cost, wrt=classifier.w)
        g_b = ts.grad(cost=cost, wrt=classifier.b)
        updates = [(classifier.w, classifier.w - arm['learning_rate'] * g_w),
                   (classifier.b, classifier.b - arm['learning_rate'] * g_b)]

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
        patience = 5000
        patience_increase = 2
        threshold = 0.995
        valid_freq = min(n_batches_train, patience // 2)

        best_valid_loss = 1.
        best_iter = 0
        test_score = 1.
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
                        with open(arm['dir'] + '/best_model.pkl', 'wb') as file:
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
                   ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

        return train_loss, best_valid_loss, test_score, current_track

    def get_search_space(self):
        params = {
            'learning_rate': Param('learning_rate', np.log(1 * 10 ** (-3)), np.log(1 * 10 ** (-1)), dist='uniform',
                                   scale='log'),
            'batch_size': Param('batch_size', 1, 1000, dist='uniform', scale='linear', interval=1)
        }

        return params
