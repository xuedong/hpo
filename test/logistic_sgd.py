import numpy as np
import timeit
import theano
import theano.tensor as T

from six.moves import cPickle

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

    # early-stopping parameters
    patience = 5000
    patience_increase = 2
    threshold = 0.995
    valid_freq = min(n_batches_train, patience // 2)

    best_valid_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done = False
    epoch = 0
    while (epoch < epochs) and not done:
        epoch += 1
        for batch_index in range(n_batches_train):
            batch_cost = train_model(batch_index)
            iter = (epoch - 1) * n_batches_train + batch_index

            if (iter + 1) % valid_freq == 0:
                valid_losses = [valid_model(i) for i in range(n_batches_valid)]
                current_valid_loss = np.mean(valid_losses)

                print(
                    'epoch %i, batch %i/%i, batch average cost %f, validation error %f %%' %
                    (
                        epoch,
                        batch_index + 1,
                        n_batches_train,
                        batch_cost,
                        current_valid_loss * 100.)
                )

                if current_valid_loss < best_valid_loss:
                    if current_valid_loss < best_valid_loss * threshold:
                        patience = max(patience, iter * patience_increase)

                    best_valid_loss = current_valid_loss

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
                    with open('best_model_logistic_sgd.pkl', 'wb') as file:
                        cPickle.dump(classifier, file)

            if patience <= iter:
                done = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization completed with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_valid_loss * 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

    return ()


if __name__=='__main__':
    sgd(DATASET, LEARNING_RATE, EPOCHS, BATCH_SIZE)