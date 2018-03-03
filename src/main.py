import numpy
import sys
import os
import random
import timeit
import theano.tensor as ts

import logger
import utils
import logistic
import hyperband_finite
from params import Param


def get_search_space():
    params = {
        'learning_rate': Param('learning_rate', numpy.log(1 * 10 ** (-3)), numpy.log(1 * 10 ** (-1)), dist='uniform',
                               scale='log'),
        'batch_size': Param('batch_size', 1, 1000, dist='uniform', scale='linear', interval=1)
    }

    return params


def main():
    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)

    output_dir = ''
    random.seed(1234)
    model_name = 'logistic_sgd_'

    for seed_id in range(10):
        start_time = timeit.default_timer()

        director = output_dir + '../result/' + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir)

        x = ts.matrix('x')
        model = logistic.LogisticRegression(x, 28*28, 10)
        params = get_search_space()
        # arms = model.generate_arms(1, "../result/", params, True)
        # train_loss, val_err, test_err = logistic.run_solver(1000, arms[0], data)

        hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=3)
        hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=3, s_run=0)
        hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=3, s_run=1)
        hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=3, s_run=2)
        hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=3, s_run=3)
        hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=4, s_run=4)
        # print(train_loss, val_acc, test_acc)

        end_time = timeit.default_timer()
        print(('The code for the trial number ' +
               str(seed_id) +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)


if __name__ == "__main__":
    main()
