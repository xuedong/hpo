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
import classifiers.logistic as logistic
# import classifiers.mlp as mlp
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

        hyperband_finite.hyperband_finite(test_model, 'epochs', params, 1, 1000, 360, director, data, eta=4,
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
        exp_name = 'tpe_' + model + '_0/'
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'tpe')

        start_time = timeit.default_timer()

        trials = Trials()

        f_target = target.HyperLogistic(test_model, epochs, director, data)
        objective = f_target.objective

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
        exp_name = 'hoo_' + model + '_0/'
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'hoo')

        start_time = timeit.default_timer()

        f_target = target.TheanoLogistic(1, data, director)
        bbox = utils_ho.std_box(f_target, None, 2, 0.1,
                                [(params['learning_rate'].get_min(), params['learning_rate'].get_max()),
                                 (params['batch_size'].get_min(), params['batch_size'].get_max())],
                                [params['learning_rate'].get_type(), params['batch_size'].get_type()])

        alpha = math.log(horizon) * (sigma ** 2)
        losses = utils_ho.loss_hoo(bbox=bbox, rho=rho, nu=nu, alpha=alpha, sigma=sigma,
                                   horizon=epochs*horizon, update=False)
        losses = np.array(losses)

        with open(director + '/results.pkl', 'wb') as file:
            cPickle.dump(-losses, file)

        end_time = timeit.default_timer()

        print(('The code for the trial number ' +
               str(seed_id) +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)

        print('<-- Running HCT -->')
        exp_name = 'hct_' + model + '_0/'
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'hct')

        start_time = timeit.default_timer()

        f_target = target.TheanoLogistic(1, data, director)
        bbox = utils_ho.std_box(f_target, None, 2, 0.1,
                                [(params['learning_rate'].get_min(), params['learning_rate'].get_max()),
                                 (params['batch_size'].get_min(), params['batch_size'].get_max())],
                                [params['learning_rate'].get_type(), params['batch_size'].get_type()])

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
        exp_name = 'random_' + model + '_0/'
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'random')

        start_time = timeit.default_timer()

        best, results, track = random_search.random_search(test_model, horizon,
                                                           director, params, epochs, data, rng, verbose=False)
        cPickle.dump([best, results, track], open(director + '/results.pkl', 'wb'))

        end_time = timeit.default_timer()

        print(('The code for the trial number ' +
               str(seed_id) +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)


if __name__ == "__main__":
    main('logistic', 5, 0.66, 1., 0.1, 0.05, 25, 1000)
    # main('mlp')
