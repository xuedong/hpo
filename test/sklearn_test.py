# import numpy as np
import sys
# import os
import math
import timeit
from six.moves import cPickle

from hyperopt import tpe
from hyperopt import fmin
from hyperopt import Trials

import log.logger as logger
import source.target as target
# import source.utils as utils
import source.hyperband.hyperband_finite as hyperband_finite
import source.heuristics.hyperloop as hyperloop
import bo.tpe_hyperopt as tpe_hyperopt
import baseline.random_search as random_search
import ho.utils_ho as utils_ho
from source.classifiers.sklearn.svm_sklearn import *
from source.classifiers.sklearn.ada_sklearn import *
from source.classifiers.sklearn.gbm_sklearn import *
from source.classifiers.sklearn.knn_sklearn import *
from source.classifiers.sklearn.mlp_sklearn import *
# from source.classifiers.rf_sklearn import *
# from source.classifiers.tree_sklearn import *


if __name__ == '__main__':
    horizon = 1000
    iterations = 1
    mcmc = 1
    rhomax = 20
    rho = 0.66
    nu = 1.
    sigma = 0.1
    delta = 0.05
    alpha = math.log(horizon) * (sigma ** 2)
    c = 2 * math.sqrt(1. / (1 - 0.66))
    c1 = (0.66 / (3 * 1.)) ** (1. / 8)

    models = [SVM]
    model_names = ['svm_']
    targets = [target.SklearnMLP]
    targets_tpe = [target.HyperSKMLP]
    params_ho = [d_svm]
    # models = [SVM, Ada, GBM, KNN, MLP]
    # model_names = ['svm_', 'ada_', 'gbm_', 'knn_', 'sk_mlp_']
    # targets = [target.SklearnSVM, target.SklearnAda, target.SklearnGBM, target.SklearnKNN, target.SklearnMLP]
    # targets_tpe = [target.HyperSVM, target.HyperAda, target.HyperGBM, target.HyperKNN, target.HyperSKMLP]
    # params_ho = [d_svm, d_ada, d_gbm, d_knn, d_mlp]
    output_dir = ''
    # rng = np.random.RandomState(12345)

    path = os.path.join(os.getcwd(), '../data/uci')
    dataset = 'breast_cancer.csv'
    problem = 'binary'
    target_index = 0
    data = utils.build(os.path.join(path, dataset), target_index)
    x, y = data

    for i in range(len(models)):
        model = models[i]
        test_model = model()
        params = model.get_search_space()
        for seed_id in range(mcmc):
            # print('<-- Running Hyperloop -->')
            # exp_name = 'hyperloop_' + model_names[i] + '2/'
            # director = output_dir + '../result/' + exp_name + model_names[i] + str(seed_id)
            # if not os.path.exists(director):
            #     os.makedirs(director)
            # log_dir = output_dir + '../log/' + exp_name + model_names[i] + str(seed_id)
            # if not os.path.exists(log_dir):
            #     os.makedirs(log_dir)
            # sys.stdout = logger.Logger(log_dir, 'hyperloop')
            #
            # start_time = timeit.default_timer()
            #
            # hyperloop.hyperloop_finite(test_model, 'iterations', params, 1, 10, 360, director, data,
            #                            eta=4, verbose=True)
            # # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 1000, 360, director, data, eta=4,
            # # s_run=0, verbose=False)
            # # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4,
            # # s_run=1, verbose=True)
            # # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4,
            # # s_run=2, verbose=True)
            # # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4,
            # # s_run=3, verbose=True)
            # # hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=4, s_run=4)
            #
            # end_time = timeit.default_timer()
            #
            # print(('The code for the trial number ' +
            #        str(seed_id) +
            #        ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)
            #
            # print('<-- Running Hyperband -->')
            # exp_name = 'hyperband_' + model_names[i] + '2/'
            # director = output_dir + '../result/' + exp_name + model_names[i] + str(seed_id)
            # if not os.path.exists(director):
            #     os.makedirs(director)
            # log_dir = output_dir + '../log/' + exp_name + model_names[i] + str(seed_id)
            # if not os.path.exists(log_dir):
            #     os.makedirs(log_dir)
            # sys.stdout = logger.Logger(log_dir, 'hyperband')
            #
            # start_time = timeit.default_timer()
            #
            # hyperband_finite.hyperband_finite(test_model, 'iterations', params, 1, 10, 360, director, data,
            #                                   eta=4, verbose=True)
            # # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 1000, 360, director, data, eta=4,
            # # s_run=0, verbose=False)
            # # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4,
            # # s_run=1, verbose=True)
            # # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4,
            # # s_run=2, verbose=True)
            # # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4,
            # # s_run=3, verbose=True)
            # # hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=4, s_run=4)
            #
            # end_time = timeit.default_timer()
            #
            # print(('The code for the trial number ' +
            #        str(seed_id) +
            #        ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

            print('<-- Running TPE -->')
            exp_name = 'tpe_' + model_names[i] + '2/'
            director = output_dir + '../result/' + exp_name + model_names[i] + str(seed_id)
            if not os.path.exists(director):
                os.makedirs(director)
            log_dir = output_dir + '../log/' + exp_name + model_names[i] + str(seed_id)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            sys.stdout = logger.Logger(log_dir, 'tpe')

            start_time = timeit.default_timer()

            trials = Trials()

            # f_target_tpe = target.HyperLogistic(test_model, epochs, director, data)
            f_target_tpe = targets_tpe[i](test_model, iterations, director, data)
            objective = f_target_tpe.objective

            best = fmin(objective,
                        space=tpe_hyperopt.convert_params(params),
                        algo=tpe.suggest,
                        max_evals=horizon,
                        trials=trials)

            with open(director + '/results.pkl', 'wb') as file:
                cPickle.dump([trials, best], file)

            end_time = timeit.default_timer()

            print(('The code for the trial number ' +
                   str(seed_id) +
                   ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

            print('<-- Running POO -->')
            exp_name = 'poo_' + model_names[i] + '2/'
            director = output_dir + '../result/' + exp_name + model_names[i] + str(seed_id)
            if not os.path.exists(director):
                os.makedirs(director)
            log_dir = output_dir + '../log/' + exp_name + model_names[i] + str(seed_id)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            sys.stdout = logger.Logger(log_dir, 'poo')

            start_time = timeit.default_timer()

            rhos = [float(j) / float(rhomax) for j in range(1, rhomax + 1)]
            f_target = targets[i](test_model, x, y, '5fold', problem, director)
            bbox = utils_ho.std_box(f_target, None, 3, 0.1,
                                    [params_ho[i][key][1] for key in params_ho[i].keys()],
                                    [params_ho[i][key][0] for key in params_ho[i].keys()])
            losses = utils_ho.loss_poo(bbox=bbox, rhos=rhos, nu=nu, alpha=alpha, sigma=sigma,
                                       horizon=horizon, director=director)
            losses = np.array(losses)

            with open(director + '/results.pkl', 'wb') as file:
                cPickle.dump(-losses, file)

            end_time = timeit.default_timer()

            print(('The code for the trial number ' +
                   str(seed_id) +
                   ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

            print('<-- Running PCT -->')
            exp_name = 'pct_' + model_names[i] + '2/'
            director = output_dir + '../result/' + exp_name + model_names[i] + str(seed_id)
            if not os.path.exists(director):
                os.makedirs(director)
            log_dir = output_dir + '../log/' + exp_name + model_names[i] + str(seed_id)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            sys.stdout = logger.Logger(log_dir, 'pct')

            start_time = timeit.default_timer()

            rhos = [float(j) / float(rhomax) for j in range(1, rhomax + 1)]
            f_target = targets[i](test_model, x, y, '5fold', problem, director)
            bbox = utils_ho.std_box(f_target, None, 3, 0.1,
                                    [params_ho[i][key][1] for key in params_ho[i].keys()],
                                    [params_ho[i][key][0] for key in params_ho[i].keys()])
            losses = utils_ho.loss_pct(bbox=bbox, rhos=rhos, nu=nu, c=c, c1=c1, delta=delta,
                                       horizon=horizon, director=director)
            losses = np.array(losses)

            with open(director + '/results.pkl', 'wb') as file:
                cPickle.dump(-losses, file)

            end_time = timeit.default_timer()

            print('<-- Running Random Search -->', )
            exp_name = 'random_' + model_names[i] + '2/'
            director = output_dir + '../result/' + exp_name + model_names[i] + str(seed_id)
            if not os.path.exists(director):
                os.makedirs(director)
            log_dir = output_dir + '../log/' + exp_name + model_names[i] + str(seed_id)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            sys.stdout = logger.Logger(log_dir, 'random')

            start_time = timeit.default_timer()

            best, results, track_valid, track_test = random_search.random_search(test_model, 'iterations', horizon,
                                                                                 director, params,
                                                                                 1, data, verbose=True)
            cPickle.dump([best, results, track_valid, track_test], open(director + '/results.pkl', 'wb'))

            end_time = timeit.default_timer()

            print(('The code for the trial number ' +
                   str(seed_id) +
                   ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)
