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