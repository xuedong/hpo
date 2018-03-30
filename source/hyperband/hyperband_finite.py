import numpy as np
import timeit
import os

from six.moves import cPickle

import utils


def sh_finite(model, resource_type, params, n, i, eta, big_r, director, data,
              rng=np.random.RandomState(1234), track=np.array([1.]), verbose=False):
    """Successive halving.

    :param model: model to be trained
    :param resource_type: type of resource to be allocated
    :param params: hyperparameter search space
    :param n: number of configurations in this successive halving phase
    :param i: the number of the bracket
    :param eta: elimination proportion
    :param big_r: number of resources
    :param director: where we store the results
    :param data: dataset
    :param rng: random state
    :param track: initial track vector
    :param verbose: verbose option
    :return: the dictionary of arms, the stored results and the vector of test errors
    """
    arms = model.generate_arms(n, director, params)
    remaining_arms = [list(a) for a in
                      zip(arms.keys(), [0] * len(arms.keys()), [0] * len(arms.keys()), [0] * len(arms.keys()))]
    current_track = np.copy(track)

    for l in range(i+1):
        num_pulls = int(big_r * eta ** (l - i))
        num_arms = int(n * eta ** (-l))
        print('%d\t%d' % (num_arms, num_pulls))
        for a in range(len(remaining_arms)):
            start_time = timeit.default_timer()
            arm_key = remaining_arms[a][0]

            if verbose:
                print(arms[arm_key])

            if not os.path.exists(arms[arm_key]['dir'] + '/best_model.pkl'):
                train_loss, val_err, test_err, current_track = \
                    model.run_solver(num_pulls, arms[arm_key], data,
                                     rng=rng, track=current_track, verbose=verbose)
            else:
                classifier = cPickle.load(open(arms[arm_key]['dir'] + '/best_model.pkl', 'rb'))
                train_loss, val_err, test_err, current_track = \
                    model.run_solver(num_pulls, arms[arm_key], data,
                                     rng=rng, classifier=classifier, track=current_track, verbose=verbose)

            if verbose:
                print(arm_key, train_loss, val_err, test_err, utils.s_to_m(start_time, timeit.default_timer()))

            arms[arm_key]['results'].append([num_pulls, train_loss, val_err, test_err])
            remaining_arms[a][1] = train_loss
            remaining_arms[a][2] = val_err
            remaining_arms[a][3] = test_err
        remaining_arms = sorted(remaining_arms, key=lambda a: a[2])
        n_k1 = int(n * eta ** (-l-1))
        if i-l-1 >= 0:
            # for k in range(n_k1, len(remaining_arms)):
                # arm_dir = arms[remaining_arms[k][0]]['dir']
                # files = os.listdir(arm_dir)
            remaining_arms = remaining_arms[0:n_k1]
    best_arm = arms[remaining_arms[0][0]]

    return arms, [best_arm, remaining_arms[0][1], remaining_arms[0][2], remaining_arms[0][3]], current_track


def hyperband_finite(model, resource_type, params, min_units, max_units, runtime, director, data,
                     rng=np.random.RandomState(1234), eta=4., budget=0, n_hyperbands=1,
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
    :param n_hyperbands: maximum number of hyperbands to run
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
        best_val = 1.
        track = np.array([1.])
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
                    arms, result, track = \
                        sh_finite(model, resource_type, params, n, i, eta, big_r, director,
                                  rng=rng, data=data, track=track, verbose=verbose)
                    results[(k, s)] = arms
                    if verbose:
                        print("k = " + str(k) + ", lscale = " + str(s) + ", validation error = " + str(
                            result[2]) + ", test error = " + str(
                            result[3]) + " best arm dir: " + result[0]['dir'])
                    durations.append([utils.s_to_m(start_time, timeit.default_timer()), result])
                    print("time elapsed: " + str(utils.s_to_m(start_time, timeit.default_timer())))

                    if result[2] < best_val:
                        best_val = result[2]
                        # best_n = n
                        # best_i = i
                        # best_arm = result[0]

                if s_run is None:
                    cPickle.dump([durations, results, track], open(director + '/results.pkl', 'wb'))
                else:
                    cPickle.dump([durations, results, track],
                                 open(director + '/results_' + str(s_run) + '.pkl', 'wb'))
                s -= 1
