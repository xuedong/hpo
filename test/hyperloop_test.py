import numpy as np
import sys
import os
# import math
# import random
import timeit
# from six.moves import cPickle

import log.logger as logger
import utils

# import target
import classifiers.logistic as logistic
import classifiers.mlp as mlp
# import hyperband.hyperband_finite as hyperband_finite
from heuristics.hyperloop import hyperloop_finite


def main(model, mcmc):
    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)

    output_dir = ''
    rng = np.random.RandomState(12345)
    model_name = model + '_sgd_'

    # test_model = mlp.MLP
    # params = mlp.MLP.get_search_space()
    test_model = logistic.LogisticRegression
    params = logistic.LogisticRegression.get_search_space()

    exp_id = 1

    for seed_id in range(9, mcmc):
        print('<-- Running Hyperloop -->')
        exp_name = 'hyperloop_' + model + '_' + str(exp_id) + '/'
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'hyperloop')

        start_time = timeit.default_timer()

        hyperloop_finite(test_model, 'epochs', params, 1, 100, 360, director, data, eta=4, verbose=True)
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


if __name__ == "__main__":
    main('logistic', 10)
    # main('mlp')
