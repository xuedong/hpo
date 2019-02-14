import os
import gzip
import theano
import theano.tensor as ts
import numpy as np
import pandas as pd

from six.moves import urllib
from six.moves import cPickle

from sklearn.metrics import log_loss, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


def load_data(dataset):
    """Function that loads the data at dataset.

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
        except FileNotFoundError:
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


def s_to_m(start_time, current_time):
    """Function that converts time in seconds to time in minutes.

    :param start_time: starting time in seconds
    :param current_time: current time in seconds
    :return: minutes
    """
    return (current_time - start_time) / 60.


def log_eta(x, eta):
    """Compute log of x with base eta.

    :param x: input value
    :param eta: base
    :return: rounded log_eta(x)
    """
    return np.round(np.log(x) / np.log(eta), decimals=10)


def get_budget(min_units, max_units, eta):
    """Compute the total budget.

    :param min_units: minimum units of resources can be allocated to one configuration
    :param max_units: maximum units of resources can be allocated to one configuration
    :param eta: elimination proportion
    :return: the corresponding total budget and the number of configurations
    """
    budget = int(np.floor(log_eta(max_units / min_units, eta)) + 1) * max_units
    return budget, budget / max_units


class Loss:
    def __init__(self, model, x, y, method='holdout', problem='binary'):
        self.model = model
        self.x = x
        self.y = y
        self.method = method
        self.problem = problem
        sc = StandardScaler()
        self.x = sc.fit_transform(self.x)
        if self.problem == 'binary':
            # self.loss = log_loss
            self.loss = accuracy_score
        elif self.problem == 'cont':
            self.loss = mean_squared_error
        else:
            self.loss = log_loss

    def evaluate_loss(self, **param):
        if self.method == 'holdout':
            x_train_valid, x_test, y_train_valid, y_test = train_test_split(self.x, self.y, test_size=0.2)
            x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=0.2)
            clf = self.model.__class__(problem=self.problem, **param).eval()
            clf.fit(x_train, y_train)
            if self.problem == 'binary':
                y_hat_valid = clf.predict(x_valid)
                y_hat_test = clf.predict(x_test)
            elif self.problem == 'cont':
                y_hat_test = clf.predict(x_test)
                y_hat_valid = clf.predict(x_valid)
            else:
                y_hat_test = clf.predict_proba(x_test)
                y_hat_valid = clf.predict_proba(x_valid)
            valid_error = self.loss(y_valid, y_hat_valid)
            test_error = self.loss(y_test, y_hat_test)
            return valid_error, test_error
        elif self.method == '5fold':
            kf = KFold(n_splits=5, shuffle=True)
            losses = []
            test_errors = []
            x_train_valid, x_test, y_train_valid, y_test = train_test_split(self.x, self.y, test_size=0.2)
            for train_index, valid_index in kf.split(x_train_valid):
                x_train, x_valid = x_train_valid[train_index], x_train_valid[valid_index]
                y_train, y_valid = y_train_valid[train_index], y_train_valid[valid_index]
                clf = self.model.__class__(problem=self.problem, **param).eval()
                clf.fit(x_train, y_train)
                if self.problem == 'binary':
                    y_hat_valid = clf.predict(x_valid)
                    y_hat_test = clf.predict(x_test)
                elif self.problem == 'cont':
                    y_hat_valid = clf.predict(x_valid)
                    y_hat_test = clf.predict(x_test)
                else:
                    y_hat_valid = clf.predict_proba(x_valid)
                    y_hat_test = clf.predict_proba(x_test)
                loss = self.loss(y_valid, y_hat_valid)
                # if loss > 1:
                #     print(y_valid)
                #     print(y_hat_valid)
                # print(y_hat_valid)
                losses.append(loss)
                test_errors.append(self.loss(y_test, y_hat_test))
            # print(losses)
            return np.average(losses), np.average(test_errors)


def cum_max(history):
    n = len(history)
    res = np.empty((n,))
    for i in range(n):
        res[i] = np.max(history[:(i + 1)])
    return res


def build(csv_path, target_index, header=None):
    data = pd.read_csv(csv_path, header=header)
    data = data.as_matrix()
    y = data[:, target_index]
    x = np.delete(data, obj=np.array([target_index]), axis=1)
    return x, y


def second_largest(arr):
    count = 0
    m1 = m2 = float('-inf')
    length = len(arr)
    for i in range(length):
        count += 1
        if arr[i] > m2:
            if arr[i] >= m1:
                m1, m2 = arr[i], m1
            else:
                m2 = arr[i]

    return m2, arr.index(m2) if count >= 2 else None
