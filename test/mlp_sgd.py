import numpy as np
import os
import timeit
import theano
import theano.tensor as ts

from six.moves import cPickle

import sys

import src.classifiers.mlp as mlp
import src.utils as utils

# Parameters
LEARNING_RATE = 0.13
L1_REG = 0.
L2_REG = 0.0001
EPOCHS = 1000
BATCH_SIZE = 20
HIDDEN = 500

rng = np.random.RandomState(1234)

# Dataset
DATASET = 'mnist.pkl.gz'


def sgd(dataset, learning_rate, l1_reg, l2_reg, epochs, batch_size, n_hidden):
    """Applying stochastic gradient descent on a single mlp.

    :param dataset: path to the dataset
    :param learning_rate: learning rate used
    :param l1_reg: L1-norm regularization constant
    :param l2_reg: L2-norm regularization constant
    :param epochs: number of times to run the optimizer
    :param batch_size: size of the minibatch
    :param n_hidden: number of hidden units
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

    # construct the classifier
    classifier = mlp.MLP(rng=rng, input_data=x, n_in=28*28, n_hidden=n_hidden, n_out=10)
    cost = classifier.neg_log_likelihood(y) + l1_reg * classifier.l1 + l2_reg * classifier.l2

    # construct a Theano function that computes the errors made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.zero_one(y),
        givens={
            x: test_input[index * batch_size: (index + 1) * batch_size],
            y: test_target[index * batch_size: (index + 1) * batch_size]
        }
    )
    valid_model = theano.function(
        inputs=[index],
        outputs=classifier.zero_one(y),
        givens={
            x: valid_input[index * batch_size: (index + 1) * batch_size],
            y: valid_target[index * batch_size: (index + 1) * batch_size]
        }
    )

    # construct a Theano function that updates the parameters of
    # the training model using stochastic gradient descent
    g_params = [ts.grad(cost=cost, wrt=param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * g_param)
        for param, g_param in zip(classifier.params, g_params)
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
    patience = 5000
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
                current_valid_loss = np.mean(valid_losses)

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
                    with open('../log/best_model_mlp_sgd.pkl', 'wb') as file:
                        cPickle.dump(classifier, file)

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


def predict(path, test_input):
    """Function that loads a trained model, then uses it to predict labels.

    :param path: path to the saved model
    :param test_input: inputs to be tested
    :return: None
    """
    # load a trained model
    classifier = cPickle.load(open(path, 'rb'))

    # construct a predict function
    predict_model = theano.function(
        inputs=[classifier.input_data],
        outputs=classifier.logistic_layer.y_pred
    )

    # test it on test_input
    tests = test_input.get_value()
    predicted_labels = predict_model(tests[:10])

    print("Predicted labels for the first 10 examples: ")
    print(predicted_labels)

    return ()


if __name__ == '__main__':
    sgd(DATASET, LEARNING_RATE, L1_REG, L2_REG, EPOCHS, BATCH_SIZE, HIDDEN)
    # test_data = utils.load_data(DATASET)
    # test_inputs, _ = test_data[2]
    # predict('../log/best_model_mlp_sgd.pkl', test_inputs)
