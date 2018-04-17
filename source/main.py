import numpy as np
import sys
import os
import math
# import random
import timeit
from six.moves import cPickle

import log.logger as logger
import utils

from hyperopt import tpe
from hyperopt import fmin
from hyperopt import Trials

import target
# import classifiers.logistic as logistic
import classifiers.mlp as mlp
import hyperband.hyperband_finite as hyperband_finite
import bo.tpe_hyperopt as tpe_hyperopt
import baseline.random_search as random_search
import ho.utils_ho as utils_ho


def main(model, mcmc, rho, nu, sigma, delta, horizon, epochs):
    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)

    output_dir = ''
    rng = np.random.RandomState(12345)
    model_name = model + '_sgd_'

    test_model = mlp.MLP
    params = mlp.MLP.get_search_space()
    # test_model = logistic.LogisticRegression
    # params = logistic.LogisticRegression.get_search_space()

    exp_id = 0

    for seed_id in range(mcmc):
        print('<-- Running Hyperband -->')
        exp_name = 'hyperband_' + model + '_' + str(exp_id) + '/'
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'hyperband')

        start_time = timeit.default_timer()

        hyperband_finite.hyperband_finite(test_model, 'epochs', params, 1, 10, 360, director, data, eta=4,
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

        print('<-- Running TPE -->')
        exp_name = 'tpe_' + model + '_' + str(exp_id) + '/'
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'tpe')

        start_time = timeit.default_timer()

        trials = Trials()

        # f_target_tpe = target.HyperLogistic(test_model, epochs, director, data)
        f_target_tpe = target.HyperMLP(test_model, epochs, director, data)
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

        print('<-- Running HOO -->')
        exp_name = 'hoo_' + model + '_' + str(exp_id) + '/'
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'hoo')

        start_time = timeit.default_timer()

        # f_target_hoo = target.TheanoHOOLogistic(epochs, data, director)
        f_target_hoo = target.TheanoHOOMLP(epochs, data, director)
        # bbox = utils_ho.std_box(f_target_hoo, None, 2, 0.1,
        #                         [(params['learning_rate'].get_min(), params['learning_rate'].get_max()),
        #                          (params['batch_size'].get_min(), params['batch_size'].get_max())],
        #                         [params['learning_rate'].get_type(), params['batch_size'].get_type()],
        #                         keep=True)
        bbox = utils_ho.std_box(f_target_hoo, None, 2, 0.1,
                                [(params['learning_rate'].get_min(), params['learning_rate'].get_max()),
                                 (params['batch_size'].get_min(), params['batch_size'].get_max()),
                                 (params['l2_reg'].get_min(), params['l2_reg'].get_max())],
                                [params['learning_rate'].get_type(), params['batch_size'].get_type(),
                                 params['l2_reg'].get_type()],
                                keep=True)

        alpha = math.log(horizon) * (sigma ** 2)
        losses = utils_ho.loss_hoo(bbox=bbox, rho=rho, nu=nu, alpha=alpha, sigma=sigma,
                                   horizon=horizon, update=False, keep=True)
        losses = np.array(losses)

        with open(director + '/results.pkl', 'wb') as file:
            cPickle.dump(losses, file)

        end_time = timeit.default_timer()

        print(('The code for the trial number ' +
               str(seed_id) +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

        print('<-- Running HCT -->')
        exp_name = 'hct_' + model + '_' + str(exp_id) + '/'
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'hct')

        start_time = timeit.default_timer()

        # f_target_hct = target.TheanoHCTLogistic(1, data, director)
        f_target_hct = target.TheanoHCTMLP(1, data, director)
        # bbox = utils_ho.std_box(f_target_hct, None, 2, 0.1,
        #                         [(params['learning_rate'].get_min(), params['learning_rate'].get_max()),
        #                          (params['batch_size'].get_min(), params['batch_size'].get_max())],
        #                         [params['learning_rate'].get_type(), params['batch_size'].get_type()])
        bbox = utils_ho.std_box(f_target_hct, None, 2, 0.1,
                                [(params['learning_rate'].get_min(), params['learning_rate'].get_max()),
                                 (params['batch_size'].get_min(), params['batch_size'].get_max()),
                                 (params['l2_reg'].get_min(), params['l2_reg'].get_max())],
                                [params['learning_rate'].get_type(), params['batch_size'].get_type(),
                                 params['l2_reg'].get_type()],
                                keep=True)

        c = 2 * math.sqrt(1. / (1 - rho))
        c1 = (rho / (3 * nu)) ** (1. / 8)
        losses = utils_ho.loss_hct(bbox=bbox, rho=rho, nu=nu, c=c, c1=c1, delta=delta, sigma=sigma,
                                   horizon=epochs*horizon, keep=True)
        losses = np.array(losses)

        with open(director + '/results.pkl', 'wb') as file:
            cPickle.dump(-losses, file)

        end_time = timeit.default_timer()

        print(('The code for the trial number ' +
               str(seed_id) +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

        print('<-- Running Random Search -->', )
        exp_name = 'random_' + model + '_' + str(exp_id) + '/'
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'random')

        start_time = timeit.default_timer()

        best, results, track_valid, track_test = random_search.random_search(test_model, horizon,
                                                                             director, params,
                                                                             epochs, data, rng, verbose=True)
        cPickle.dump([best, results, track_valid, track_test], open(director + '/results.pkl', 'wb'))

        end_time = timeit.default_timer()

        print(('The code for the trial number ' +
               str(seed_id) +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

        # TODO: write an experiment report


if __name__ == "__main__":
    main('mlp', 1, 0.66, 1., 0.1, 0.05, 4, 10)
    # main('mlp')
