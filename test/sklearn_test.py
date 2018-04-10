import numpy as np
import sys
import os
import timeit

import log.logger as logger
import source.utils as utils
import source.hyperband.hyperband_finite as hyperband_finite
from source.classifiers.svm_sklearn import *
from source.classifiers.ada_sklearn import *
from source.classifiers.gbm_sklearn import *
from source.classifiers.knn_sklearn import *
from source.classifiers.mlp_sklearn import *
from source.classifiers.rf_sklearn import *
from source.classifiers.tree_sklearn import *


if __name__ == '__main__':
    models = [SVM]
    model_names = ['svm_']
    # models = [SVM, Ada, GBM, KNN, MLP, RF, Tree]
    # model_names = ['svm_', 'ada_', 'gbm_', 'knn_', 'sk_mlp_', 'rf_', 'tree_']
    output_dir = ''
    # rng = np.random.RandomState(12345)

    path = os.path.join(os.getcwd(), '../data/uci')
    dataset = 'wine.csv'
    problem = 'cont'
    target_index = 0
    data = utils.build(os.path.join(path, dataset), target_index)

    for i in range(len(models)):
        model = models[i]
        test_model = model()
        params = model.get_search_space()
        for seed_id in range(1):
            print('<-- Running Hyperband -->')
            exp_name = 'hyperband_' + model_names[i] + '1/'
            director = output_dir + '../result/' + exp_name + model_names[i] + str(seed_id)
            if not os.path.exists(director):
                os.makedirs(director)
            log_dir = output_dir + '../log/' + exp_name + model_names[i] + str(seed_id)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            sys.stdout = logger.Logger(log_dir, 'hyperband')

            start_time = timeit.default_timer()

            hyperband_finite.hyperband_finite(test_model, 'iterations', params, 1, 10, 360, director, data, eta=4,
                                              verbose=True)
            # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 1000, 360, director, data, eta=4,
            # s_run=0, verbose=False)
            # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4,
            # s_run=1, verbose=True)
            # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4,
            # s_run=2, verbose=True)
            # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4,
            # s_run=3, verbose=True)
            # hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=4, s_run=4)

            end_time = timeit.default_timer()

            print(('The code for the trial number ' +
                   str(seed_id) +
                   ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)
