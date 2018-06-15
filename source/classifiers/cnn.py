from __future__ import print_function

import sys
import timeit
import os
import numpy as np

from six.moves import cPickle

import theano
import theano.tensor as ts
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from models import Model
from params import Param
from classifiers.logistic import LogisticRegression
from classifiers.mlp import MLP


class ConvolutionPoolLayer(object):
    def __init__(self, rng, input_data, filter_shape, image_shape, pool_size):
        """This layer includes one convolution layer followed by a max-pooling layer.

        :param rng: random state
        :param input_data: input image tensor
        :param filter_shape: (nb of filters, nb of input feature maps, filter height, filter width)
        :param image_shape: (batch size, nb of input feature maps, image height, image width)
        :param pool_size: pooling/subsampling factors
        """
        assert image_shape[1] == filter_shape[1]
        self.input_data = input_data
        # initialize (nb of input feature maps) * (nb of filters) weight matrices
        # where the bounds depend on the input and output dimensions
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:])) // np.prod(pool_size)
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        w_values = np.asarray(
            rng.uniform(
                low=-w_bound,
                high=w_bound,
                size=filter_shape
            ),
            dtype=theano.config.floatX
        )
        self.w = theano.shared(value=w_values, borrow=True)
        # initialize (nb of filers) biases
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        # output after convolution and pooling
        # convolution between input feature maps and filters
        convolution = conv2d(
            input=input_data,
            filters=self.w,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        # max-pooling for each feature map
        pooling = pool.pool_2d(
            input=convolution,
            ws=pool_size,
            ignore_border=True
        )
        self.output = ts.tanh(pooling + self.b.dimshuffle('x', 0, 'x', 'x'))
        # parameters of the model
        self.params = [self.w, self.b]
        # keep track of the input data
        self.input_data = input_data


class CNN(Model):
    def __init__(self, input_data, n_in, n_hidden, n_out, rng=np.random.RandomState(1234)):
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
        # Loss functions
        self.neg_log_likelihood = self.logistic_layer.neg_log_likelihood
        self.zero_one = self.logistic_layer.zero_one
        # parameters of the model
        self.params = self.hidden_layer.params + self.logistic_layer.params
        # keep track of the input
        self.input_data = input_data

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
            arm = {'dir': path + "/" + dirname,
                   'learning_rate': 0.001, 'batch_size': 100, 'n_hidden': 500,
                   'l1_reg': 0., 'l2_reg': 1., 'results': []}
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
            hps = ['learning_rate', 'batch_size', 'l2_reg']
            for hp in hps:
                val = params[hp].get_param_range(1, stochastic=True)
                arm[hp] = val[0]
            arm['l1_reg'] = 0.
            arm['n_hidden'] = 500
            arm['results'] = []
            arms[i] = arm

        os.chdir('../../../source')

        return arms

    @staticmethod
    def run_solver(epochs, arm, data, rng=None, classifier=None,
        track_valid = np.array([1.]), track_test = np.array([1.]), verbose = False):
        """LeNet with 2 convolution-pooling layers, 1 fully-connected layer and 1 logistic layer.

        :param epochs: number of epochs
        :param arm: hyperparameter configuration encoded as a dictionary
        :param data: dataset to use
        :param rng: not used here
        :param classifier: initial model, set as None by default
        :param track_valid: vector where we store the best validation errors
        :param track_test: vector where we store the test errors
        :param verbose: verbose option
        :return:
        """
        train_input, train_target = data[0]
        valid_input, valid_target = data[1]
        test_input, test_target = data[2]

        n_batches_train = train_input.get_value(borrow=True).shape[0] // batch_size
        n_batches_valid = valid_input.get_value(borrow=True).shape[0] // batch_size
        n_batches_test = test_input.get_value(borrow=True).shape[0] // batch_size

        print('Building model...')

        # symbolic variables
        index = ts.lscalar()
        x = ts.matrix('x')
        y = ts.ivector('y')

        # construct the first convolution-pooling layer
        layer1_input = x.reshape((batch_size, 1, 28, 28))
        layer1 = cnn.ConvolutionPoolLayer(
            rng=rng,
            input_data=layer1_input,
            filter_shape=(kernels[0], 1, 5, 5),
            image_shape=(batch_size, 1, 28, 28),
            pool_size=(2, 2)
        )

        # construct the second convolution-pooling layer
        layer2 = cnn.ConvolutionPoolLayer(
            rng=rng,
            input_data=layer1.output,
            filter_shape=(kernels[1], kernels[0], 5, 5),
            image_shape=(batch_size, kernels[0], 12, 12),
            pool_size=(2, 2)
        )

        # construct the fully-connected hidden layer
        layer3_input = layer2.output.flatten(2)
        layer3 = mlp.HiddenLayer(
            rng=rng,
            input_data=layer3_input,
            n=kernels[1] * 4 * 4,
            m=n_hidden,
            activation=ts.tanh
        )

        # construct the output layer
        layer4 = logistic.LogisticRegression(
            input_data=layer3.output,
            n=n_hidden,
            m=10
        )

        # compute the cost
        cost = layer4.neg_log_likelihood(y)

        # construct a Theano function that computes the errors made
        # by the model a minibatch
        test_model = theano.function(
            inputs=[index],
            outputs=layer4.zero_one(y),
            givens={
                x: test_input[index * batch_size: (index + 1) * batch_size],
                y: test_target[index * batch_size: (index + 1) * batch_size]
            }
        )
        valid_model = theano.function(
            inputs=[index],
            outputs=layer4.zero_one(y),
            givens={
                x: valid_input[index * batch_size: (index + 1) * batch_size],
                y: valid_target[index * batch_size: (index + 1) * batch_size]
            }
        )

        # construct a Theano function that updates the parameters of
        # the training model using stochastic gradient descent
        params = layer4.params + layer3.params + layer2.params + layer1.params
        g_params = [ts.grad(cost=cost, wrt=param) for param in params]
        updates = [
            (param, param - learning_rate * g_param)
            for param, g_param in zip(params, g_params)
        ]

        train_model = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_input[index * batch_size: (index + 1) * batch_size],
                y: train_target[index * batch_size: (index + 1) * batch_size]
            }
        )

        print('Training model...')

        # early-stopping parameters
        patience = 10000
        patience_increase = 2
        threshold = 0.995
        valid_freq = min(n_batches_train, patience // 2)

        best_valid_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        done = False
        epoch = 0
        while (epoch < epochs) and not done:
            epoch += 1
            for batch_index in range(n_batches_train):
                batch_cost = train_model(batch_index)
                iteration = (epoch - 1) * n_batches_train + batch_index

                if (iteration + 1) % valid_freq == 0:
                    valid_losses = [valid_model(i) for i in range(n_batches_valid)]
                    current_valid_loss = float(np.mean(valid_losses))

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

                        test_losses = [test_model(i) for i in range(n_batches_test)]
                        test_score = np.mean(test_losses)

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
                        # with open('../log/best_model_mlp_sgd.pkl', 'wb') as file:
                        #     cPickle.dump(classifier, file)

                if patience <= iteration:
                    done = True
                    break

        end_time = timeit.default_timer()
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

        return ()

    @staticmethod
    def get_search_space():
        params = {
            'learning_rate': Param('learning_rate', np.log(1 * 10 ** (-3)), np.log(1 * 10 ** (-1)), dist='uniform',
                                   scale='log'),
            'batch_size': Param('batch_size', 1, 1000, dist='uniform', scale='linear', interval=1),
            'l2_reg': Param('l2_reg', np.log(1 * 10 ** (-4)), np.log(1 * 10 ** (-2)), dist='uniform', scale='log')
        }

        return params
