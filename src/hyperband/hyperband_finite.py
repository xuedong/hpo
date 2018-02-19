import numpy as np
import timeit
import utils

from six.moves import cPickle


def sha_finite(model, resource_type, params, n, i, eta, big_r, path):
    """

    :param model:
    :param resource_type:
    :param params:
    :param n:
    :param i:
    :param eta:
    :param big_r:
    :param path:
    :return:
    """
    


def hyperband_finite(model, resource_type, params, min_units, max_units, runtime, path, eta=4., budget=0, n_hyperbands=2, s_run=None, doubling=False):
    """Hyperband with finite horizon.

    :param model: object with subroutines to generate arms and train models
    :param resource_type: type of resource to be allocated
    :param params: hyperparameter search space
    :param min_units: minimum units of resources can be allocated to one configuration
    :param max_units: maximum units of resources can be allocated to one configuration
    :param runtime: runtime patience (in min)
    :param path: path to the directory where output are stored
    :param eta: elimination proportion
    :param budget: total budget for one bracket
    :param n_hyperbands: maximum number of hyperbands to run
    :param s_run: option to repeat a specific bracket
    :param doubling: option to decide whether we want to double the per bracket budget in the outer loop
    :return: None
    """
    start_time = timeit.default_timer()
    # result storage
    results = {}
    durations = []

    # outer loop
    k = 0
    while utils.s_to_m(start_time, timeit.default_timer()) < runtime and k < n_hyperbands:
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
        best_val = 0
        print(' s_max = %i' % s_max)

        # inner loop
        while s >= 0 and utils.s_to_m(start_time, timeit.default_timer()) < runtime:
            # specify the number of configurations
            n = int(budget/big_r * eta**s/(s+1.))

            if n > 0:
                i = 0
                while n*big_r*(i+1.)*eta**(-i) > budget:
                    i += 1

                    print('i = %d, n = %d' % (i, n))
                    arms, result = sha_finite(model, resource_type, params, n, i, eta, big_r, path)
                    results[(k, s)] = arms
                    print("k = " + str(k) + ", l = " + str(s) + ", validation accuracy = " + str(
                        result[2]) + ", test accuracy = " + str(
                        result[3]) + " best arm dir: " + result[0]['dir'])
                    durations.append([utils.s_to_m(start_time, timeit.default_timer()), result])
                    print("time elapsed: " + str(utils.s_to_m(start_time, timeit.default_timer())))

                    if result[2] > best_val:
                        best_val = result[2]
                        # best_n = n
                        # best_i = i
                        # best_arm = result[0]

                cPickle.dump([durations, results], open(path+'/results.pkl', 'w'))
                s -= 1
