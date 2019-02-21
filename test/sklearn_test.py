# import numpy as np
import sys
# import os
import math
import timeit
# import pandas as pd
from six.moves import cPickle

from hyperopt import tpe
from hyperopt import fmin
from hyperopt import Trials

import log.logger as logger
import target
# import utils
import hyperband.hyperband_finite as hyperband_finite
import heuristics.hyperloop as hyperloop
import bo.tpe_hyperopt as tpe_hyperopt
import baseline.random_search as random_search
import heuristics.dttts as dttts
import ho.utils_ho as utils_ho
from classifiers.sklearn.svm_sklearn import *
from classifiers.sklearn.ada_sklearn import *
from classifiers.sklearn.gbm_sklearn import *
from classifiers.sklearn.knn_sklearn import *
from classifiers.sklearn.mlp_sklearn import *
# from classifiers.rf_sklearn import *
# from classifiers.tree_sklearn import *


if __name__ == '__main__':
    horizon = 162
    iterations = 1
    mcmc = 100
    rhomax = 20
    rho = 0.66
    nu = 1.
    sigma = 0.1
    delta = 0.05
    alpha = math.log(horizon) * (sigma ** 2)
    # c = 2 * math.sqrt(1. / (1 - 0.66))
    # c1 = (0.66 / (3 * 1.)) ** (1. / 8)
    verbose = False

    models = [SVM]
    model_names = ['svm_']
    targets = [target.SklearnSVM]
    targets_tpe = [target.HyperSVM]
    params_ho = [d_svm]
    # models = [Ada, GBM, KNN]
    # model_names = ['ada_', 'gbm_', 'knn_']
    # targets = [target.SklearnAda, target.SklearnGBM, target.SklearnKNN]
    # targets_tpe = [target.HyperAda, target.HyperGBM, target.HyperKNN]
    # params_ho = [d_ada, d_gbm, d_knn]
    output_dir = ''
    # rng = np.random.RandomState(12345)

    methods = {"hyperloop": True, "hyperband": True, "gpo": True, "tpe": True, "random": True, "dttts": True}
    # methods = {"hyperloop": True, "hyperband": False, "gpo": False, "tpe": True, "random": True, "dttts": True}

    path = os.path.join(os.getcwd(), '../data/uci')
    # dataset = 'adult.csv'
    # problem = 'binary'
    # target_index = 88
    # data = utils.sub_build(os.path.join(path, dataset), target_index, frac=0.3)
    dataset = 'adult_train.csv'
    testset = 'adult_test.csv'
    problem = 'binary'
    target_index = 88

    # build dataset and test set
    test = utils.sub_build(os.path.join(path, testset), target_index=target_index, frac=1)
    x_test, y_test = test
    data = utils.sub_build(os.path.join(path, dataset), target_index=target_index, frac=0.1)
    # data = utils.build(os.path.join(path, dataset), target_index=target_index)
    x, y = data
    exp_index = 3

    for i in range(len(models)):
        model = models[i]
        test_model = model()
        params = model.get_search_space()
        for seed_id in range(mcmc):
            if methods["hyperloop"]:
                print('<-- Running Hyperloop -->')
                exp_name = 'hyperloop_' + model_names[i] + str(exp_index) + '/'
                director = output_dir + '../result/' + exp_name + model_names[i] + str(seed_id)
                if not os.path.exists(director):
                    os.makedirs(director)
                log_dir = output_dir + '../log/' + exp_name + model_names[i] + str(seed_id)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                sys.stdout = logger.Logger(log_dir, 'hyperloop')

                start_time = timeit.default_timer()

                hyperloop.hyperloop_finite(test_model, 'iterations', params, 1, 18, 360, director, data, test,
                                           eta=3, problem=problem, verbose=True)
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

            if methods["hyperband"]:
                print('<-- Running Hyperband -->')
                exp_name = 'hyperband_' + model_names[i] + str(exp_index) + '/'
                director = output_dir + '../result/' + exp_name + model_names[i] + str(seed_id)
                if not os.path.exists(director):
                    os.makedirs(director)
                log_dir = output_dir + '../log/' + exp_name + model_names[i] + str(seed_id)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                sys.stdout = logger.Logger(log_dir, 'hyperband')

                start_time = timeit.default_timer()

                hyperband_finite.hyperband_finite(test_model, 'iterations', params, 1, 18, 360, director, data, test,
                                                  eta=3, problem=problem, verbose=verbose)
                # hyperband_finite.hyperband_finite(test_model, 'iterations', params, 1, 10, 360, director, data, eta=4,
                #                                   s_run=0, verbose=True)
                # hyperband_finite.hyperband_finite(test_model, 'iterations', params, 1, 32, 360, director, data, eta=4,
                #                                   s_run=1, verbose=True)
                # hyperband_finite.hyperband_finite(test_model, 'iterations', params, 1, 24, 360, director, data, eta=2,
                #                                   s_run=2, verbose=True)
                # hyperband_finite.hyperband_finite(test_model, 'iterations', params, 1, 32, 360, director, data, eta=4,
                #                                   s_run=3, verbose=True)
                # hyperband_finite.hyperband_finite(model, 'iterations', params, 1, 24, 360, director, data, eta=2,
                #                                   s_run=4, verbose=True)

                end_time = timeit.default_timer()

                print(('The code for the trial number ' +
                       str(seed_id) +
                       ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

            if methods["tpe"]:
                print('<-- Running TPE -->')
                exp_name = 'tpe_' + model_names[i] + str(exp_index) + '/'
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
                f_target_tpe = targets_tpe[i](test_model, iterations, director, problem, data, test)
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

            if methods["gpo"]:
                print('<-- Running GPO -->')
                exp_name = 'gpo_' + model_names[i] + str(exp_index) + '/'
                director = output_dir + '../result/' + exp_name + model_names[i] + str(seed_id)
                if not os.path.exists(director):
                    os.makedirs(director)
                log_dir = output_dir + '../log/' + exp_name + model_names[i] + str(seed_id)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                sys.stdout = logger.Logger(log_dir, 'gpo')

                start_time = timeit.default_timer()

                rhos = [float(j) / float(rhomax) for j in range(1, rhomax + 1)]
                f_target = targets[i](test_model, x, y, x_test, y_test, '5fold', problem, director)
                bbox = utils_ho.std_box(f_target, None, 3, 0.1,
                                        [params_ho[i][key][1] for key in params_ho[i].keys()],
                                        [params_ho[i][key][0] for key in params_ho[i].keys()])
                valid_losses, test_losses = utils_ho.loss_poo(bbox=bbox, rhos=rhos, nu=nu, alpha=alpha, sigma=0,
                                                              horizon=horizon*iterations, director=director)
                valid_losses = np.array(valid_losses)
                test_losses = np.array(test_losses)

                with open(director + '/results.pkl', 'wb') as file:
                    cPickle.dump([-valid_losses, -test_losses], file)

                end_time = timeit.default_timer()

                print(('The code for the trial number ' +
                       str(seed_id) +
                       ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

                # if methods["pct"]:
                #     print('<-- Running PCT -->')
                #     exp_name = 'pct_' + model_names[i] + str(exp_index) + '/'
                #     director = output_dir + '../result/' + exp_name + model_names[i] + str(seed_id)
                #     if not os.path.exists(director):
                #         os.makedirs(director)
                #     log_dir = output_dir + '../log/' + exp_name + model_names[i] + str(seed_id)
                #     if not os.path.exists(log_dir):
                #         os.makedirs(log_dir)
                #     sys.stdout = logger.Logger(log_dir, 'pct')
                #
                #     start_time = timeit.default_timer()
                #
                #     rhos = [float(j) / float(rhomax) for j in range(1, rhomax + 1)]
                #     f_target = targets[i](test_model, x, y, '5fold', problem, director)
                #     bbox = utils_ho.std_box(f_target, None, 3, 0.1,
                #                             [params_ho[i][key][1] for key in params_ho[i].keys()],
                #                             [params_ho[i][key][0] for key in params_ho[i].keys()])
                #     losses = utils_ho.loss_pct(bbox=bbox, rhos=rhos, nu=nu, c=c, c1=c1, delta=delta,
                #                                horizon=horizon, director=director)
                #     losses = np.array(losses)
                #
                #     with open(director + '/results.pkl', 'wb') as file:
                #         cPickle.dump(-losses, file)
                #
                #     end_time = timeit.default_timer()

            if methods["random"]:
                print('<-- Running Random Search -->', )
                exp_name = 'random_' + model_names[i] + str(exp_index) + '/'
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
                                                                                     iterations, data, test,
                                                                                     problem=problem, verbose=False)
                cPickle.dump([best, results, track_valid, track_test], open(director + '/results.pkl', 'wb'))

                end_time = timeit.default_timer()

                print(('The code for the trial number ' +
                       str(seed_id) +
                       ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

            if methods["dttts"]:
                print('<-- Running Dynamic TTTS -->', )
                exp_name = 'dttts_' + model_names[i] + str(exp_index) + '/'
                director = output_dir + '../result/' + exp_name + model_names[i] + str(seed_id)
                if not os.path.exists(director):
                    os.makedirs(director)
                log_dir = output_dir + '../log/' + exp_name + model_names[i] + str(seed_id)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                sys.stdout = logger.Logger(log_dir, 'dttts')

                start_time = timeit.default_timer()

                best, results, track_valid, track_test = dttts.dttts(test_model, 'iterations', params,
                                                                     horizon*iterations, 2, horizon*iterations,
                                                                     director, data, test,
                                                                     problem=problem, verbose=False)
                cPickle.dump([best, results, track_valid, track_test], open(director + '/results.pkl', 'wb'))

                end_time = timeit.default_timer()

                print(('The code for the trial number ' +
                       str(seed_id) +
                       ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)
