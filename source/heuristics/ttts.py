import numpy as np
import timeit
# import os

from scipy.stats import beta
from scipy.stats import bernoulli
from six.moves import cPickle

import utils


def ttts(model, resource_type, params, n, i, budget, director, data, frac=0.5, dist='Bernoulli',
         rng=np.random.RandomState(12345), track_valid=np.array([1.]), track_test=np.array([1.]),
         problem='cont', verbose=False):
    """Top-Two Thompson Sampling.

    :param model: model to be trained
    :param resource_type: type of resource to be allocated
    :param params: hyperparameter search space
    :param n: number of configurations in this ttts phase
    :param i: the number of the bracket
    :param budget: number of resources
    :param director: where we store the results
    :param data: dataset
    :param frac: threshold in ttts
    :param dist: type of prior distribution
    :param rng: random state
    :param track_valid: initial track vector
    :param track_test: initial track vector
    :param problem: type of problem (classification or regression)
    :param verbose: verbose option
    :return: the dictionary of arms, the stored results and the vector of test errors
    """
    arms = model.generate_arms(n, director, params)
    remaining_arms = []
    if resource_type == 'epochs':
        remaining_arms = [list(a) for a in
                          zip(arms.keys(), [0] * len(arms.keys()), [0] * len(arms.keys()), [0] * len(arms.keys()))]
    elif resource_type == 'iterations':
        remaining_arms = [list(a) for a in zip(arms.keys(), [0] * len(arms.keys()), [0] * len(arms.keys()))]
    current_track_valid = np.copy(track_valid)
    current_track_test = np.copy(track_test)

    succ = np.zeros(n)
    fail = np.zeros(n)
    num_pulls = np.zeros(n)
    rewards = np.zeros(n)
    # means = np.zeros(n)

    start_time = timeit.default_timer()
    for a in range(n):
        arm_key = remaining_arms[a][0]
        num_pulls[a] = 1
        if resource_type == 'epochs':
            train_loss, val_err, test_err, current_track_valid, current_track_test = \
                model.run_solver(1, arms[arm_key], data, rng=rng,
                                 track_valid=current_track_valid, track_test=current_track_test, verbose=verbose)
            rewards[a] = val_err
        elif resource_type == 'iterations':
            val_err, avg_loss, current_track_valid, current_track_test = \
                model.run_solver(1, arms[arm_key], data,
                                 rng=rng, track_valid=current_track_valid,
                                 track_test=current_track_test, problem=problem, verbose=verbose)
            rewards[a] = 1 + avg_loss

    # best = 0
    for _ in range(n, int(budget)):
        # means = rewards / num_pulls
        # best = np.random.choice(np.flatnonzero(means == means.max()))

        # print(rewards)
        ts = np.zeros(n)
        for a in range(n):
            if dist == 'Bernoulli':
                alpha_prior = 1
                beta_prior = 1
                trial = bernoulli.rvs(1-rewards[a])
                if trial == 1:
                    succ[a] += 1
                else:
                    fail[a] += 1
                ts[a] = beta.rvs(alpha_prior + succ[a], beta_prior + fail[a], size=1)[0]

        idx_i = np.argmax(ts)
        # print(idx_i)
        # print("\n"+str(ts[idx_i])+"\n")
        if np.random.rand() > frac:
            idx_j = idx_i
            threshold = 10000
            count = 0
            while idx_i == idx_j and count < threshold:
                ts = np.zeros(n)
                if dist == 'Bernoulli':
                    alpha_prior = 1
                    beta_prior = 1
                    for a in range(n):
                        trial = bernoulli.rvs(1 - rewards[a])
                        if trial == 1:
                            succ[a] += 1
                        else:
                            fail[a] += 1
                        ts[a] = beta.rvs(alpha_prior + succ[a], beta_prior + fail[a], size=1)[0]
                idx_j = np.argmax(ts)
                count += 1
                # print(str(idx_j)+": "+str(ts[idx_j]))
            if idx_i != idx_j:
                idx_i = idx_j
            else:
                _, idx_j = utils.second_largest(list(ts))
                idx_i = idx_j

        if resource_type == 'epochs':
            arm_key = remaining_arms[int(idx_i)][0]
            classifier = cPickle.load(open(arms[arm_key]['dir'] + '/best_model.pkl', 'rb'))
            train_loss, val_err, test_err, current_track_valid, current_track_test = \
                model.run_solver(1, arms[arm_key], data,
                                 rng=rng, classifier=classifier,
                                 track_valid=current_track_valid, track_test=current_track_test,
                                 verbose=verbose)
            rewards[idx_i] = val_err
            num_pulls[idx_i] += 1

            if verbose:
                print(arm_key, train_loss, val_err, test_err, utils.s_to_m(start_time, timeit.default_timer()))

            arms[arm_key]['results'].append([num_pulls[idx_i], train_loss, val_err, test_err])
            remaining_arms[int(idx_i)][1] = train_loss
            remaining_arms[int(idx_i)][2] = val_err
            remaining_arms[int(idx_i)][3] = test_err
        elif resource_type == 'iterations':
            arm_key = remaining_arms[int(idx_i)][0]
            val_err, avg_loss, current_track_valid, current_track_test = \
                model.run_solver(1, arms[arm_key], data,
                                 rng=rng, track_valid=current_track_valid,
                                 track_test=current_track_test, verbose=verbose)
            rewards[idx_i] = 1 + avg_loss
            num_pulls[idx_i] += 1

            if verbose:
                print(arm_key, val_err, utils.s_to_m(start_time, timeit.default_timer()))

            arms[arm_key]['results'].append([num_pulls[idx_i], val_err, avg_loss])
            remaining_arms[int(idx_i)][1] = -val_err
            remaining_arms[int(idx_i)][2] = -avg_loss

    if resource_type == 'epochs':
        remaining_arms = sorted(remaining_arms, key=lambda a: a[2])
    elif resource_type == 'iterations':
        remaining_arms = sorted(remaining_arms, key=lambda a: a[2])

    best_arm = arms[remaining_arms[0][0]]

    result = []
    if resource_type == 'epochs':
        result = [best_arm, remaining_arms[0][1], remaining_arms[0][2], remaining_arms[0][3]]
    elif resource_type == 'iterations':
        result = [best_arm, remaining_arms[0][1], remaining_arms[0][2]]

    return arms, result, current_track_valid, current_track_test
