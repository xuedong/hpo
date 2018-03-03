import os
import numpy as np
import matplotlib.pyplot as plt

from six.moves import cPickle


def plot_hyperband(path, s_max, trials):
    """

    :param path:
    :param s_max:
    :param trials:
    :return:
    """
    os.chdir(path)
    fig = plt.figure()
    longest = 0

    for s in range(s_max+1):
        tracks = np.array([None for _ in range(trials)])
        for i in range(trials):
            [_, _, track] = cPickle.load(open(path + 'logistic_sgd_' + str(i) + '/results_' + str(s) + '.pkl', 'rb'))
            tracks[i] = track

        length = len(tracks[0])
        if length > longest:
            longest = length
        x = range(length)
        y = np.mean(tracks, axis=0)
        if s == 0:
            plt.plot(x, y, label=r"Random Search")
        else:
            plt.plot(x, y, label=r"$\mathtt{s=}")

    tracks = np.array([None for _ in range(trials)])
    for i in range(trials):
        [_, _, track] = cPickle.load(open(path + 'logistic_sgd_' + str(i) + '/results.pkl', 'rb'))
        tracks[i] = track[0:longest]

    # length = len(tracks[0])
    x = range(longest)
    y = np.mean(tracks, axis=0)
    plt.plot(x, y, label=r"Hyperband")

    plt.grid()
    plt.legend(loc=0)
    plt.show()
