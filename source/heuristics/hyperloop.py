import numpy as np
import timeit
# import os

from six.moves import cPickle

import utils
from heuristics.ttts import ttts


def hyperloop_finite(model, resource_type, params, min_units, max_units, runtime, director, data,
                     rng=np.random.RandomState(1234), eta=4., budget=0, n_hyperloops=1,
                     s_run=None, doubling=False, verbose=False):
    """Hyperband with finite horizon.

    :param model: object with subroutines to generate arms and train models
    :param resource_type: type of resource to be allocated
    :param params: hyperparameter search space
    :param min_units: minimum units of resources can be allocated to one configuration
    :param max_units: maximum units of resources can be allocated to one configuration
    :param runtime: runtime patience (in min)
    :param director: path to the directory where output are stored
    :param data: dataset to use
    :param rng: random state
    :param eta: elimination proportion
    :param budget: total budget for one bracket
    :param n_hyperloops: maximum number of hyperloops to run
    :param s_run: option to repeat a specific bracket
    :param doubling: option to decide whether we want to double the per bracket budget in the outer loop
    :param verbose: verbose option
    :return: None
    """
    start_time = timeit.default_timer()
    # result storage
    results = {}
    durations = []

    # outer loop
    k = 0
    while utils.s_to_m(start_time, timeit.default_timer()) < runtime and k < n_hyperloops:
        # initialize the budget according to whether we do doubling trick or not
        if budget == 0:
            if not doubling:
                budget = int(np.floor(utils.log_eta(max_units/min_units, eta)) + 1) * max_units
            else:
                budget = int((2 ** k) * max_units)

        k += 1

        print('\nBudget B = %i' % budget)
        print('##################')

        big_r = float(max_units)
        r = float(min_units)
        s_max = int(min(budget / big_r - 1, int(np.floor(utils.log_eta(big_r / r, eta)))))
        s = s_max
        best_val = 1.
        track_valid = np.array([1.])
        track_test = np.array([1.])
        print('s_max = %i' % s_max)

        # inner loop
        while s >= 0 and utils.s_to_m(start_time, timeit.default_timer()) < runtime:
            # specify the number of configurations
            n = int(budget/big_r * eta**s/(s+1.))

            if n > 0:
                i = 0
                while n*big_r*(i+1.)*eta**(-i) > budget:
                    i += 1

                if s_run is None or i == s_run:
                    print('s = %d, n = %d' % (i, n))
                    arms, result, track_valid, track_test = \
                        ttts(model, resource_type, params, n, i, big_r, director,
                             rng=rng, data=data, track_valid=track_valid, track_test=track_test, verbose=verbose)
                    results[(k, s)] = arms

                    if resource_type == 'epochs':
                        if verbose:
                            print("k = " + str(k) + ", lscale = " + str(s) + ", validation error = " + str(
                                result[2]) + ", test error = " + str(
                                result[3]) + ", best arm dir: " + result[0]['dir'])

                        if result[2] < best_val:
                            best_val = result[2]
                            # best_n = n
                            # best_i = i
                            # best_arm = result[0]
                    elif resource_type == 'iterations':
                        if verbose:
                            print("k = " + str(k) + ", lscale = " + str(s) + ", validation error = " + str(
                                result[1]) + ", best arm dir: " + result[0]['dir'])

                        if result[1] < best_val:
                            best_val = result[1]

                    durations.append([utils.s_to_m(start_time, timeit.default_timer()), result])
                    print("time elapsed: " + str(utils.s_to_m(start_time, timeit.default_timer())))

                if s_run is None:
                    cPickle.dump([durations, results, track_valid, track_test], open(director + '/results.pkl', 'wb'))
                else:
                    cPickle.dump([durations, results, track_valid, track_test],
                                 open(director + '/results_' + str(s_run) + '.pkl', 'wb'))
                s -= 1
