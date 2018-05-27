import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle

from bo.tpe_hyperopt import combine_tracks


def plot_random(path, trials, classifier_name, optimizer_name, dataset_name, idx):
    """Plot test error evaluation of random search.

    :param path:
    :param trials:
    :param classifier_name:
    :param optimizer_name:
    :param dataset_name:
    :param idx:
    :return:
    """
    os.chdir(path)
    fig = plt.figure()

    tracks = np.array([None for _ in range(trials)])
    shortest = sys.maxsize
    for i in range(trials):
        [_, _, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i)
                                             + '/results.pkl', 'rb'))
        if len(track) < shortest:
            shortest = len(track)
    for i in range(trials):
        [_, _, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i)
                                             + '/results.pkl', 'rb'))
        tracks[i] = track[0:shortest]

    length = len(tracks[0])
    x = range(length)
    y = np.mean(tracks, axis=0)
    plt.plot(x, y, label=r"Random Search")

    plt.grid()
    plt.ylim((0, 0.2))
    plt.legend(loc=0)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Epochs')
    save_path = os.path.join(os.path.abspath('../../'), 'img/{}'.format('random_' + classifier_name + str(idx)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(os.path.abspath('../../'), 'img/{}/{}.pdf'.format('random_' + classifier_name +
                                                                               str(idx), dataset_name)))
    plt.close(fig)


def plot_hyperband_only(path, trials, classifier_name, optimizer_name, dataset_name, idx):
    """Plot test error evaluation of hyperband.

    :param path: path to which the result image is stored
    :param trials: number of trials of one algorithm
    :param classifier_name: name of the classifier
    :param optimizer_name: name of the optimizer
    :param dataset_name: name of the dataset
    :param idx: id of the experiment
    :return:
    """
    os.chdir(path)
    fig = plt.figure()
    shortest = sys.maxsize

    tracks = np.array([None for _ in range(trials)])
    for i in range(trials):
        [_, _, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        if len(track) < shortest:
            shortest = len(track)
    for i in range(trials):
        [_, _, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        tracks[i] = track[0:shortest]

    # length = len(tracks[0])
    x = range(shortest)
    y = np.mean(tracks, axis=0)
    plt.plot(x, y, label=r"Hyperband")

    plt.grid()
    # plt.ylim((0, 0.2))
    plt.legend(loc=0)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Evaluations')
    save_path = os.path.join(os.path.abspath('../../'), 'img/{}'.format('hyperband_' + classifier_name + str(idx)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(os.path.abspath('../../'), 'img/{}/{}.pdf'.format('hyperband_' + classifier_name +
                                                                               str(idx), dataset_name)))

    os.chdir('..')

    plt.close(fig)


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


def plot_hyperloop_only(path, trials, classifier_name, optimizer_name, dataset_name, idx):
    """Plot test error evaluation of hyperloop.

    :param path: path to which the result image is stored
    :param trials: number of trials of one algorithm
    :param classifier_name: name of the classifier
    :param optimizer_name: name of the optimizer
    :param dataset_name: name of the dataset
    :param idx: id of the experiment
    :return:
    """
    os.chdir(path)
    fig = plt.figure()
    shortest = sys.maxsize

    tracks = np.array([None for _ in range(trials)])
    for i in range(trials):
        [_, _, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        if len(track) < shortest:
            shortest = len(track)
    for i in range(trials):
        [_, _, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        tracks[i] = track[0:shortest]

    # length = len(tracks[0])
    x = range(shortest)
    y = np.mean(tracks, axis=0)
    plt.plot(x, y, label=r"Hyperloop")

    plt.grid()
    # plt.ylim((0, 0.2))
    plt.legend(loc=0)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Evaluations')
    save_path = os.path.join(os.path.abspath('../../'), 'img/{}'.format('hyperloop_' + classifier_name + str(idx)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(os.path.abspath('../../'), 'img/{}/{}.pdf'.format('hyperloop_' + classifier_name +
                                                                               str(idx), dataset_name)))

    os.chdir('..')

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
    # if devs:
    #     err = np.std(tracks, axis=0)
    #     lower = y - err
    #     higher = y + err
    #     plt.fill_between(x, lower, higher, facecolor='lightblue')
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


def plot_hoo(path, runs, classifier_name, optimizer_name, dataset_name, idx):
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

    losses = np.array([None for _ in range(runs)])
    for i in range(runs):
        loss = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        if len(loss) < shortest:
            shortest = len(loss)
    for i in range(runs):
        loss = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        losses[i] = loss[0:shortest]

    # length = len(tracks[0])
    x = range(shortest+1)
    y = np.mean(losses, axis=0)
    y = np.append([1.], y)
    # if devs:
    #     err = np.std(tracks, axis=0)
    #     lower = y - err
    #     higher = y + err
    #     plt.fill_between(x, lower, higher, facecolor='lightblue')
    plt.plot(x, y, label=r"HOO")

    plt.grid()
    plt.ylim((0, 1))
    plt.legend(loc=0)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Evaluations')
    save_path = os.path.join(os.path.abspath('../../'), 'img/{}'.format('hoo_' + classifier_name + str(idx)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(os.path.abspath('../../'), 'img/{}/{}.pdf'.format('hoo_' + classifier_name +
                                                                               str(idx), dataset_name)))

    os.chdir('..')

    plt.close(fig)


def plot_hct(path, runs, classifier_name, optimizer_name, dataset_name, idx):
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

    losses = np.array([None for _ in range(runs)])
    for i in range(runs):
        loss = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        if len(loss) < shortest:
            shortest = len(loss)
    for i in range(runs):
        loss = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        losses[i] = loss[0:shortest]

    # length = len(tracks[0])
    x = range(shortest + 1)
    y = np.mean(losses, axis=0)
    y = np.append([1.], y)
    # if devs:
    #     err = np.std(tracks, axis=0)
    #     lower = y - err
    #     higher = y + err
    #     plt.fill_between(x, lower, higher, facecolor='lightblue')
    plt.plot(x, y, label=r"HCT")

    plt.grid()
    plt.ylim((0, 1))
    plt.legend(loc=0)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Evaluations')
    save_path = os.path.join(os.path.abspath('../../'), 'img/{}'.format('hct_' + classifier_name + str(idx)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(os.path.abspath('../../'), 'img/{}/{}.pdf'.format('hct_' + classifier_name +
                                                                               str(idx), dataset_name)))

    os.chdir('..')

    plt.close(fig)


def plot_bo(bo_ei_history, bo_ucb_history, random, dataset_name, model, problem):
    x = np.arange(1, len(random) + 1)
    fig = plt.figure()
    plt.plot(x, -random, label='Random search', color='red')
    plt.plot(x, -bo_ei_history, label='EI', color='blue')
    # plt.plot(x, -bo_pi_history, label='PI', color='cyan')
    plt.plot(x, -bo_ucb_history, label=r'GPUCB ($\beta=.5$)', color='yellow')
    # plt.plot(x, -bo_ucb2_history, label=r'GPUCB ($\beta=1.5$)', color='green')
    plt.grid()
    plt.legend(loc=0)
    plt.xlabel('Number of evaluations')
    if problem == 'binary':
        plt.ylabel('Log-Loss')
    else:
        plt.ylabel('MSE')
    dataset_name = dataset_name.split('.')[0]
    plt.savefig(os.path.join(os.path.abspath('.'), 'testing/results/{}/{}.pdf'.format(model.name, dataset_name)))
    plt.close(fig)


def plot_all(paths, runs, classifier_name, optimizer_name, dataset_name, idx, resource_type,
             type_plot='linear', devs=False):
    """

    :param paths:
    :param runs:
    :param classifier_name:
    :param optimizer_name:
    :param dataset_name:
    :param idx:
    :param resource_type:
    :param type_plot:
    :param devs:
    :return:
    """
    os.chdir(paths[0])
    fig = plt.figure()
    shortest = sys.maxsize

    # Hyperband
    tracks = np.array([None for _ in range(runs)])
    for i in range(runs):
        [_, _, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        if len(track) < shortest:
            shortest = len(track)
    for i in range(runs):
        [_, _, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        tracks[i] = track[0:shortest]

    # length = len(tracks[0])
    x = range(shortest)
    y = np.mean(tracks, axis=0)
    if devs:
        err = np.std(tracks, axis=0)
        lower = y - err
        higher = y + err
        plt.fill_between(x, lower, higher, alpha=0.5)
    if type_plot == 'linear':
        plt.plot(x, y, label=r"Hyperband")
    elif type_plot == 'log':
        plt.loglog(x, y, label=r"Hyperband")

    os.chdir('..')

    # TPE
    os.chdir(paths[1])
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
    if devs:
        err = np.std(tracks, axis=0)
        lower = y - err
        higher = y + err
        plt.fill_between(x, lower, higher, alpha=0.5)
    if type_plot == 'linear':
        plt.plot(x, y, label=r"TPE")
    elif type_plot == 'log':
        plt.loglog(x, y, label=r"TPE")

    os.chdir('..')

    # HOO
    os.chdir(paths[2])
    shortest = sys.maxsize

    losses = np.array([None for _ in range(runs)])
    for i in range(runs):
        loss = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        if len(loss) < shortest:
            shortest = len(loss)
    for i in range(runs):
        loss = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        losses[i] = loss[0:shortest]

    # length = len(tracks[0])
    x = range(shortest+1)
    y = np.mean(losses, axis=0)
    y = np.append([1.], y)
    if devs:
        err = np.std(tracks, axis=0)
        lower = y - err
        higher = y + err
        plt.fill_between(x, lower, higher, facecolor='lightblue')
    if type_plot == 'linear':
        plt.plot(x, y, label=r"HOO")
    elif type_plot == 'log':
        plt.loglog(x, y, label=r"HOO")

    os.chdir('..')

    # HCT
    os.chdir(paths[3])
    shortest = sys.maxsize

    losses = np.array([None for _ in range(runs)])
    for i in range(runs):
        loss = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        if len(loss) < shortest:
            shortest = len(loss)
    for i in range(runs):
        loss = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        losses[i] = loss[0:shortest]

    # length = len(tracks[0])
    x = range(shortest+1)
    y = np.mean(losses, axis=0)
    y = np.append([1.], y)
    if devs:
        err = np.std(tracks, axis=0)
        lower = y - err
        higher = y + err
        plt.fill_between(x, lower, higher, facecolor='lightblue')
    if type_plot == 'linear':
        plt.plot(x, y, label=r"HCT")
    elif type_plot == 'log':
        plt.loglog(x, y, label=r"HCT")

    os.chdir('..')

    # Random Search
    os.chdir(paths[4])
    shortest = sys.maxsize

    tracks = np.array([None for _ in range(runs)])
    for i in range(runs):
        [_, _, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i)
                                             + '/results.pkl', 'rb'))
        if len(track) < shortest:
            shortest = len(track)
    for i in range(runs):
        [_, _, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i)
                                             + '/results.pkl', 'rb'))
        tracks[i] = track[0:shortest]

    length = len(tracks[0])
    x = range(length)
    y = np.mean(tracks, axis=0)
    if devs:
        err = np.std(tracks, axis=0)
        lower = y - err
        higher = y + err
        plt.fill_between(x, lower, higher, facecolor='lightblue')
    if type_plot == 'linear':
        plt.plot(x, y, label=r"Random Search")
    elif type_plot == 'log':
        plt.loglog(x, y, label=r"Random Search")

    os.chdir('..')

    # Hyperloop
    os.chdir(paths[5])
    shortest = sys.maxsize

    tracks = np.array([None for _ in range(runs)])
    for i in range(runs):
        [_, _, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        if len(track) < shortest:
            shortest = len(track)
            print(shortest)
    for i in range(runs):
        [_, _, _, track] = cPickle.load(open(classifier_name + optimizer_name + str(i) + '/results.pkl', 'rb'))
        tracks[i] = track[0:shortest]

    # length = len(tracks[0])
    x = range(shortest)
    y = np.mean(tracks, axis=0)
    if devs:
        err = np.std(tracks, axis=0)
        lower = y - err
        higher = y + err
        plt.fill_between(x, lower, higher, alpha=0.5)
    if type_plot == 'linear':
        plt.plot(x, y, label=r"Heuristic")
    elif type_plot == 'log':
        plt.loglog(x, y, label=r"Heuristic")

    os.chdir('..')

    plt.grid()
    plt.xlim((0, 600))
    plt.ylim((0, 0.2))
    plt.legend(loc=0)
    plt.ylabel('Test Error')
    if resource_type == 'epochs':
        plt.xlabel('Number of Epochs')
    elif resource_type == 'iterations':
        plt.xlabel('Number of Iterations')
    save_path = os.path.join(os.path.abspath('../'), 'img/{}'.format(classifier_name + str(idx)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(os.path.abspath('../'), 'img/{}/{}.pdf'.format(classifier_name + str(idx), dataset_name)))
    plt.close(fig)
