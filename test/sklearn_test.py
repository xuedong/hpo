import numpy as np
import sys
import os
import timeit

import log.logger as logger
import source.utils as utils
import source.hyperband.hyperband_finite as hyperband_finite
from source.classifiers.svm_sklearn import *


if __name__ == '__main__':
    output_dir = ''
    # rng = np.random.RandomState(12345)
    model_name = 'svm_'

    model = SVM()
    path = os.path.join(os.getcwd(), '../data/uci')
    dataset = 'wine.csv'
    problem = 'cont'
    target_index = 0
    data = utils.build(os.path.join(path, dataset), target_index)
    test_model = SVM()
    params = SVM.get_search_space()

    for seed_id in range(1):
        print('<-- Running Hyperband -->')
        exp_name = 'hyperband_svm_0/'
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'hyperband')

        start_time = timeit.default_timer()

        hyperband_finite.hyperband_finite(test_model, 'iterations', params, 1, 100, 360, director, data, eta=4,
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
