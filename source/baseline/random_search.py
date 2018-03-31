import numpy as np
import timeit
import os

from six.moves import cPickle

import source.utils as utils


def random_search(model, n, director, params, num_pulls, data,
                  rng=np.random.RandomState(12345), track=np.array([1.]), verbose=False):
    """Random search for HPO.

    :param model:
    :param n:
    :param director:
    :param params:
    :param num_pulls:
    :param data:
    :param rng:
    :param track:
    :param verbose:
    :return:
    """
    arms = model.generate_arms(n, director, params)
    list_arms = [list(a) for a in
                 zip(arms.keys(), [0] * len(arms.keys()), [0] * len(arms.keys()), [0] * len(arms.keys()))]
    current_track = np.copy(track)

    print('%d\t%d' % (n, num_pulls))
    for a in range(len(list_arms)):
        start_time = timeit.default_timer()
        arm_key = list_arms[a][0]

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
        list_arms[a][1] = train_loss
        list_arms[a][2] = val_err
        list_arms[a][3] = test_err
    list_arms = sorted(list_arms, key=lambda a: a[2])
    best_arm = arms[list_arms[0][0]]

    return arms, [best_arm, list_arms[0][1], list_arms[0][2], list_arms[0][3]], current_track
