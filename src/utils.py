import os
import gzip
import theano
import theano.tensor as ts
import numpy as np

from six.moves import urllib
from six.moves import cPickle


def load_data(dataset):
    """

    :param dataset: path to the dataset
    :return: separated training, validation and test dataset
    """
    # download the dataset if not already present (mnist dataset by default)
    data_dir, data_file = os.path.split(dataset)

    if data_dir == '' and not os.path.isfile(dataset):
        # check if dataset is in the data folder
        path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(path) or data_file == 'mnist.pkl.gz':
            dataset = path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('Loading data...')

    # load the dataset
    with gzip.open(dataset, 'rb') as file:
        try:
            train, valid, test = cPickle.load(file, encoding='latin1')
        except:
            train, valid, test = cPickle.load(file)

    def shared_dataset(data, borrow=True):
        """Load dataset into shared variables.

        :param data: dataset
        :param borrow: True to permit returning of an object aliased to internal memory
        :return: shared_input and shared_target
        """
        data_input, data_target = data
        shared_input = theano.shared(np.asarray(data_input, dtype=theano.config.floatX), borrow=borrow)
        shared_target = theano.shared(np.asarray(data_target, dtype=theano.config.floatX), borrow=borrow)
        return shared_input, ts.cast(shared_target, 'int32')

    train_input, train_target = shared_dataset(train)
    valid_input, valid_target = shared_dataset(valid)
    test_input, test_target = shared_dataset(test)

    data_values = [(train_input, train_target), (valid_input, valid_target), (test_input, test_target)]

    return data_values
