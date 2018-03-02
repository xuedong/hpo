import os
import matplotlib.pyplot as plt

from six.moves import cPickle


def plot_hyperband(path, s_max):
    os.chdir(path)
    fig = plt.figure()

    for s in range(s_max+1):
        [_, _, track] = cPickle.load(open('results_' + str(s) + '.pkl', 'rb'))
        length = len(track)
        x = range(length)
        y = track
        plt.plot(x, y, label=r"$\mathtt{s=}")

    plt.grid()
    plt.legend(loc=0)
    plt.show()
