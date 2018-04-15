import numpy as np
import os
import timeit
import theano
import theano.tensor as ts
import matplotlib.pyplot as plt

# from six.moves import cPickle

import sys

import source.classifiers.logistic as logistic
import source.utils as utils

# Dataset
DATASET = 'mnist.pkl.gz'


def sgd(dataset, learning_rate, epochs, batch_size, classifier=None):
    """Applying stochastic gradient descent on a logistic regression model.

    :param dataset: path to the dataset
    :param learning_rate: learning rate used
    :param epochs: number of times to run the optimizer
    :param batch_size: size of the minibatch
    :param classifier: initial model
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
    # x = ts.matrix('x')
    y = ts.ivector('y')

    # construct the classifier
    if not classifier:
        x = ts.matrix('x')
        classifier = logistic.LogisticRegression(input_data=x, n=28*28, m=10)
    else:
        x = classifier.input_data
    cost = classifier.neg_log_likelihood(y)

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
    g_w = ts.grad(cost=cost, wrt=classifier.w)
    g_b = ts.grad(cost=cost, wrt=classifier.b)
    updates = [(classifier.w, classifier.w - learning_rate * g_w),
               (classifier.b, classifier.b - learning_rate * g_b)]

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
    track_test = np.array([])
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
                # with open('best_model_logistic_sgd.pkl', 'wb') as file:
                #     cPickle.dump(classifier, file)

            # if patience <= iteration:
            #    done = True
            #    break

        track_test = np.append(track_test, test_score)

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
           ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

    return track_test


if __name__ == '__main__':
    params = logistic.LogisticRegression.get_search_space()
    arms = logistic.LogisticRegression.generate_arms(10, './.cache', params)
    tracks = np.zeros(1000)
    for arm in arms:
        track = sgd(DATASET, arms[arm]['learning_rate'], 1000, arms[arm]['batch_size'])
        tracks = tracks + track
    n_epochs = range(len(tracks))
    test_errors = tracks / 1000.
    plt.plot(n_epochs, test_errors)

    plt.grid()
    # plt.ylim((0, 0.2))
    plt.legend(loc=0)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Epochs')
    plt.show()
