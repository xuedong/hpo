import numpy as np
import timeit
import theano
import theano.tensor as T

import sys
sys.path.insert(0, '../src/classifiers')
sys.path.insert(0, '../src')

import logistic
import utils

# Parameters
LEARNING_RATE = 0.13
EPOCHS = 1000
BATCH_SIZE = 600

# Dataset
DATASET = 'mnist.pkl.gz'


def sgd(dataset, learning_rate, epochs, batch_size):
    """Applying stochastic gradient descent on a logistic regression model.

    :param dataset: path to the dataset
    :param learning_rate: learning rate used
    :param epochs: number of times to run the optimizer
    :param batch_size: size of the minibatch
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
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    # construct the classifier
    classifier = logistic.LogisticRegression(input=x, n=28*28, m=10)
    cost = classifier.neg_log_likelihood(y)

    # construct a Theano function that computes the errors made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.zero_one(y),
        givens={
            x: test_input[index * batch_size, (index + 1) * batch_size],
            y: test_target[index * batch_size, (index + 1) * batch_size]
        }
    )
    valid_model = theano.function(
        inputs=[index],
        outputs=classifier.zero_one(y),
        givens={
            x: valid_input[index * batch_size, (index + 1) * batch_size],
            y: valid_target[index * batch_size, (index + 1) * batch_size]
        }
    )

    # construct a Theano function that updates the parameters of
    # the training model using stochastic gradient descent
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_input[index * batch_size, (index + 1) * batch_size],
            y: train_target[index * batch_size, (index + 1) * batch_size]
        }
    )

    print('Training model...')



    return ()