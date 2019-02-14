import numpy as np
import timeit
import os
import progressbar

from six.moves import cPickle

import source.utils as utils


def random_search(model, resource_type, n, director, params, num_pulls, data,
                  rng=np.random.RandomState(12345),
                  track_valid=np.array([1.]), track_test=np.array([1.]),
                  problem='cont', verbose=False):
    """Random search for HPO.

    :param model:
    :param resource_type:
    :param n:
    :param director:
    :param params:
    :param num_pulls:
    :param data:
    :param rng:
    :param track_valid:
    :param track_test:
    :param problem:
    :param verbose:
    :return:
    """
    arms = model.generate_arms(n, director, params)
    list_arms = []
    if resource_type == 'epochs':
        list_arms = [list(a) for a in
                     zip(arms.keys(), [0] * len(arms.keys()), [0] * len(arms.keys()), [0] * len(arms.keys()))]
    elif resource_type == 'iterations':
        list_arms = [list(a) for a in zip(arms.keys(), [0] * len(arms.keys()), [0] * len(arms.keys()))]
    current_track_valid = np.copy(track_valid)
    current_track_test = np.copy(track_test)

    print('%d\t%d' % (n, num_pulls))
    bar = progressbar.ProgressBar()

    for a in range(len(list_arms)):
        start_time = timeit.default_timer()
        arm_key = list_arms[a][0]

        if verbose:
            print(arms[arm_key])

        if resource_type == 'epochs':
            if not os.path.exists(arms[arm_key]['dir'] + '/best_model.pkl'):
                train_loss, val_err, test_err, current_track_valid, current_track_test = \
                    model.run_solver(num_pulls, arms[arm_key], data,
                                     rng=rng, track_valid=current_track_valid,
                                     track_test=current_track_test, verbose=verbose)
            else:
                classifier = cPickle.load(open(arms[arm_key]['dir'] + '/best_model.pkl', 'rb'))
                train_loss, val_err, test_err, current_track_valid, current_track_test = \
                    model.run_solver(num_pulls, arms[arm_key], data,
                                     rng=rng, classifier=classifier,
                                     track_valid=current_track_valid, track_test=current_track_test, verbose=verbose)

            if verbose:
                print(arm_key, train_loss, val_err, test_err, utils.s_to_m(start_time, timeit.default_timer()))

            arms[arm_key]['results'].append([num_pulls, train_loss, val_err, test_err])
            list_arms[a][1] = train_loss
            list_arms[a][2] = val_err
            list_arms[a][3] = test_err
        elif resource_type == 'iterations':
            val_err, avg_loss, current_track_valid, current_track_test = \
                model.run_solver(num_pulls, arms[arm_key], data,
                                 rng=rng, track_valid=current_track_valid,
                                 track_test=current_track_test, problem=problem, verbose=verbose)

            if verbose:
                print(arm_key, val_err, utils.s_to_m(start_time, timeit.default_timer()))

            arms[arm_key]['results'].append([num_pulls, val_err, avg_loss])
            list_arms[a][1] = val_err
            list_arms[a][2] = avg_loss

    if resource_type == "epochs":
        list_arms = sorted(list_arms, key=lambda a: a[2])
    elif resource_type == "iterations":
        list_arms = sorted(list_arms, key=lambda a: a[2])
    best_arm = arms[list_arms[0][0]]

    result = []
    if resource_type == "epochs":
        result = [best_arm, list_arms[0][1], list_arms[0][2], list_arms[0][3]]
    elif resource_type == "iterations":
        result = [best_arm, list_arms[0][1], list_arms[0][2]]

    return arms, result, current_track_valid, current_track_test
