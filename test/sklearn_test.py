import numpy as np
import sys
import os
import timeit
from six.moves import cPickle

import log.logger as logger
import utils

import hyperband.hyperband_finite as hyperband_finite


if __name__ == '__main__':
    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)

    output_dir = ''
    rng = np.random.RandomState(12345)
    model_name = model + '_sgd_'

    test_model = logistic.LogisticRegression
    params = logistic.LogisticRegression.get_search_space()

    for seed_id in range(3, mcmc):
        print('<-- Running Hyperband -->')
        exp_name = 'hyperband_' + model + '_0/'
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'hyperband')

        start_time = timeit.default_timer()

        hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 1000, 360, director, data, eta=4,
                                          verbose=True)
        # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 1000, 360, director, data, eta=4, s_run=0,
        #                                   verbose=False)
        # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=1,
        #                                   verbose=True)
        # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=2,
        #                                   verbose=True)
        # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=3,
        #                                   verbose=True)
        # hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=4, s_run=4)

        end_time = timeit.default_timer()

        print(('The code for the trial number ' +
               str(seed_id) +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)