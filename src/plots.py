import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from six.moves import cPickle


def plot_hyperband(path, s_max, trials, classifier_name, optimizer_name, dataset_name, idx):
    """Plot test error evaluation of hyperband with different s values.

    :param path: path to which the result image is stored
    :param s_max: maximum number of brackets
    :param trials: number of trials of one algorithm
    :param classifier_name: name of the classifier
    :param optimizer_name: name of the optimizer
    :param dataset_name: name of the dataset
    :param idx: id of the experiments
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
