import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle

from bo.tpe_hyperopt import combine_tracks


def plot_hyperband(path, s_max, trials, classifier_name, optimizer_name, dataset_name, idx):
    """Plot test error evaluation of hyperband with different s values.

    :param path: path to which the result image is stored
    :param s_max: maximum number of brackets
    :param trials: number of trials of one algorithm
    :param classifier_name: name of the classifier
    :param optimizer_name: name of the optimizer
    :param dataset_name: name of the dataset
    :param idx: id of the experiment
    :return:
    """
    os.chdir(path)
    fig = plt.figure()
    longest = 0

    for s in range(s_max+1):
        tracks = np.array([None for _ in range(trials)])
        shortest = sys.maxsize
        # compute the length of the shortest test error vector
        for i in range(trials):
            [_, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i)
                                              + '/results_' + str(s) + '.pkl', 'rb'))
            if len(track) < shortest:
                shortest = len(track)
        for i in range(trials):
            [_, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i)
                                              + '/results_' + str(s) + '.pkl', 'rb'))
            # truncate each test error vector by the length of the shortest one
            tracks[i] = track[0:shortest]

        # compute the longest truncated test error vector among all s values
        length = len(tracks[0])
        if length > longest:
            longest = length
        x = range(length)
        y = np.mean(tracks, axis=0)
        if s == 0:
            plt.plot(x, y, label=r"Random Search")
        else:
            plt.plot(x, y, label=r"$\mathtt{s=}$"+str(s))

    tracks = np.array([None for _ in range(trials)])
    for i in range(trials):
        [_, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        tracks[i] = track[0:longest]

    # length = len(tracks[0])
    x = range(longest)
    y = np.mean(tracks, axis=0)
    plt.plot(x, y, label=r"Hyperband")

    plt.grid()
    plt.ylim((0, 0.2))
    plt.legend(loc=0)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Epochs')
    save_path = os.path.join(os.path.abspath('../../'), 'img/{}'.format('hyperband_' + classifier_name + str(idx)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(os.path.abspath('../../'), 'img/{}/{}.pdf'.format('hyperband_' + classifier_name +
                                                                               str(idx), dataset_name)))
    plt.close(fig)


def plot_tpe(path, runs, classifier_name, optimizer_name, dataset_name, idx):
    """

    :param path: path to which the result image is stored
    :param runs: number of runs
    :param classifier_name: name of the classifier
    :param optimizer_name: name of the optimizer
    :param dataset_name: name of the dataset
    :param idx: id of the experiment
    :return:
    """
    os.chdir(path)
    fig = plt.figure()
    shortest = sys.maxsize

    tracks = np.array([None for _ in range(runs)])
    for i in range(runs):
        [trials, _] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        track = combine_tracks(trials)
        if len(track) < shortest:
            shortest = len(track)
    for i in range(runs):
        [trials, _] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        track = combine_tracks(trials)
        tracks[i] = track[0:shortest]

    # length = len(tracks[0])
    x = range(shortest)
    y = np.mean(tracks, axis=0)
    plt.plot(x, y, label=r"TPE")

    plt.grid()
    plt.ylim((0, 0.2))
    plt.legend(loc=0)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Epochs')
    save_path = os.path.join(os.path.abspath('../../'), 'img/{}'.format('tpe_' + classifier_name + str(idx)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(os.path.abspath('../../'), 'img/{}/{}.pdf'.format('tpe_' + classifier_name +
                                                                               str(idx), dataset_name)))
    plt.close(fig)


def plot_all(path1, path2, s_max, runs, classifier_name, optimizer_name, dataset_name, idx):
    os.chdir(path1)
    fig = plt.figure()

    # Hyperband brackets
    longest = 0
    for s in range(s_max + 1):
        tracks = np.array([None for _ in range(runs)])
        shortest = sys.maxsize
        # compute the length of the shortest test error vector
        for i in range(runs):
            [_, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i)
                                              + '/results_' + str(s) + '.pkl', 'rb'))
            if len(track) < shortest:
                shortest = len(track)
        for i in range(runs):
            [_, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i)
                                              + '/results_' + str(s) + '.pkl', 'rb'))
            # truncate each test error vector by the length of the shortest one
            tracks[i] = track[0:shortest]

        # compute the longest truncated test error vector among all s values
        length = len(tracks[0])
        if length > longest:
            longest = length
        x = range(length)
        y = np.mean(tracks, axis=0)
        if s == 0:
            plt.plot(x, y, label=r"Random Search")
        else:
            plt.plot(x, y, label=r"$\mathtt{s=}$" + str(s))

    # Hyperband
    tracks = np.array([None for _ in range(runs)])
    for i in range(runs):
        [_, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        tracks[i] = track[0:longest]

    # length = len(tracks[0])
    x = range(longest)
    y = np.mean(tracks, axis=0)
    plt.plot(x, y, label=r"Hyperband")

    # TPE
    os.chdir(path2)
    shortest = sys.maxsize

    tracks = np.array([None for _ in range(runs)])
    for i in range(runs):
        [trials, _] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        track = combine_tracks(trials)
        if len(track) < shortest:
            shortest = len(track)
    for i in range(runs):
        [trials, _] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        track = combine_tracks(trials)
        tracks[i] = track[0:shortest]

    # length = len(tracks[0])
    x = range(shortest)
    y = np.mean(tracks, axis=0)
    plt.plot(x, y, label=r"TPE")

    plt.grid()
    plt.ylim((0, 0.2))
    plt.legend(loc=0)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Epochs')
    save_path = os.path.join(os.path.abspath('../../'), 'img/{}'.format(classifier_name + str(idx)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(os.path.abspath('../../'), 'img/{}/{}.pdf'.format(classifier_name +
                                                                               str(idx), dataset_name)))
    plt.close(fig)
