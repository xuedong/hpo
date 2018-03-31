import numpy as np
import sys
import os
import timeit

import log.logger as logger
import source.utils as utils

import source.classifiers.logistic as logistic
import source.baseline.random_search as random_search


if __name__ == '__main__':
    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)

    output_dir = ''
    rng = np.random.RandomState(12345)
    model_name = 'logistic_sgd_'
    exp_name = 'random_logistic_0/'

    for seed_id in range(1):
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'random')

        test_model = logistic.LogisticRegression
        params = logistic.LogisticRegression.get_search_space()

        start_time = timeit.default_timer()

        random_search.random_search(test_model, 10, director, params, 10, data, rng, verbose=True)

        end_time = timeit.default_timer()

        print(('The code for the trial number ' +
               str(seed_id) +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)
