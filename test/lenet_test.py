import numpy as np
import os
import timeit
import theano
import theano.tensor as ts

# from six.moves import cPickle

import sys

import source.classifiers.logistic as logistic
import source.classifiers.mlp as mlp
import source.classifiers.cnn as cnn
import source.utils as utils

# Parameters
LEARNING_RATE = 0.1
L1_REG = 0.
L2_REG = 0.0001
EPOCHS = 1
BATCH_SIZE = 500
HIDDEN = 500
KERNELS = [20, 50]

rng = np.random.RandomState(1234)

# Dataset
DATASET = 'mnist.pkl.gz'


def lenet(dataset, learning_rate, epochs, batch_size, n_hidden, kernels):
    """LeNet with 2 convolution-pooling layers, 1 fully-connected layer and 1 logistic layer.

    :param dataset: path to the dataset
    :param learning_rate: learning rate used
    :param epochs: number of times to run the optimizer
    :param batch_size: size of the minibatch
    :param n_hidden: number of hidden units in the fully-connected layer
    :param kernels: number of filters for each convolution-pooling layer
    :return: None
    """
    data = utils.load_data(dataset)
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


if __name__ == '__main__':
    lenet(DATASET, LEARNING_RATE, EPOCHS, BATCH_SIZE, HIDDEN, KERNELS)
