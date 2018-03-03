import os
import numpy as np
import matplotlib.pyplot as plt

from six.moves import cPickle


def plot_hyperband(path, s_max, trials, model_name, dataset_name):
    """

    :param path: path to which the result image is stored
    :param s_max: maximum number of brackets
    :param trials: number of trials of one algorithm
    :param model_name: name of the target classifier
    :param dataset_name: name of the dataset
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
            plt.plot(x, y, label=r"$\mathtt{s=}$"+str(s))

    tracks = np.array([None for _ in range(trials)])
    for i in range(trials):
        [_, _, track] = cPickle.load(open(path + 'logistic_sgd_' + str(i) + '/results.pkl', 'rb'))
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
    save_path = os.path.join(os.path.abspath('..'), 'img/{}'.format(model_name))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(os.path.abspath('..'), 'img/{}/{}.pdf'.format(model_name, dataset_name)))
    plt.close(fig)
