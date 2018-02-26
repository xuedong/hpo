import numpy
import theano.tensor as ts

import utils
import logistic
from params import Param


def get_cnn_search_space():
    params = {}
    params['learning_rate'] = Param('learning_rate', numpy.log(1*10**(-3)), numpy.log(1*10**(-1)), dist='uniform', scale='log')
    params['batch_size'] = Param('batch_size', 1, 1000, dist='uniform', scale='linear', interval=1)

    return params


def main():
    # Use for testing
    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)
    x = ts.imatrix('x')
    model = logistic.LogisticRegression(x, 28*28, 10)
    params = get_cnn_search_space()
    arms = model.generate_arms(1, "../result/", params, True)
    train_loss, val_acc, test_acc = logistic.run_solver(1000, arms[0], data)
    print(train_loss, val_acc, test_acc)


if __name__ == "__main__":
    main()
