import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_poo(path, runs, classifier_name, optimizer_name, dataset_name, idx):
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
    plt.plot(x, y, label=r"POO(HOO)")

    plt.grid()
    # plt.ylim((0, 1))
    plt.legend(loc=0)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Evaluations')
    save_path = os.path.join(os.path.abspath('../../'), 'img/{}'.format('poo_' + classifier_name + str(idx)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(os.path.abspath('../../'), 'img/{}/{}.pdf'.format('poo_' + classifier_name +
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


def plot_pct(path, runs, classifier_name, optimizer_name, dataset_name, idx):
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
    plt.plot(x, y, label=r"POO(HCT)")

    plt.grid()
    # plt.ylim((0, 1))
    plt.legend(loc=0)
    plt.ylabel('Test Error')
    plt.xlabel('Number of Evaluations')
    save_path = os.path.join(os.path.abspath('../../'), 'img/{}'.format('pct_' + classifier_name + str(idx)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(os.path.abspath('../../'), 'img/{}/{}.pdf'.format('pct_' + classifier_name +
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


def plot_all(paths, runs_begin, runs_end, classifier_name, optimizer_name, dataset_name, idx, resource_type,
             plot='valid', devs=False, marker=False):
    """

    :param paths:
    :param runs_begin:
    :param runs_end:
    :param classifier_name:
    :param optimizer_name:
    :param dataset_name:
    :param idx:
    :param resource_type:
    :param plot:
    :param devs:
    :param marker:
    :return:
    """
    sns.set_style("darkgrid")
    os.chdir(paths[0])
    fig = plt.figure()
    shortest = sys.maxsize

    # Hyperband
    valid_tracks = np.array([None for _ in range(runs_end-runs_begin)])
    test_tracks = np.array([None for _ in range(runs_end-runs_begin)])
    for i in range(runs_end-runs_begin):
        [_, _, valid_track, _] = \
            cPickle.load(open(classifier_name + optimizer_name + str(i+runs_begin) + '/results.pkl', 'rb'))
        if len(valid_track) < shortest:
            shortest = len(valid_track)
    for i in range(runs_end-runs_begin):
        [_, _, valid_track, test_track] = \
            cPickle.load(open(classifier_name + optimizer_name + str(i+runs_begin) + '/results.pkl', 'rb'))
        valid_tracks[i] = valid_track[0:shortest]
        test_tracks[i] = test_track[0:shortest]

    # length = len(tracks[0])
    x = range(shortest)
    if resource_type == 'iterations':
        valid_tracks += 1
        test_tracks += 1
    y_valid = np.mean(valid_tracks, axis=0)
    y_test = np.mean(test_tracks, axis=0)
    if devs:
        if plot == 'valid':
            err = np.std(valid_tracks, axis=0) / 10
            lower = y_valid - err
            higher = y_valid + err
            plt.fill_between(x[1:], lower[1:], higher[1:], alpha=0.4)
    if plot == 'valid':
        if marker:
            plt.plot(x[1:], y_valid[1:], marker='.', label=r"Hyperband")
        else:
            plt.plot(x[1:], y_valid[1:], label=r"Hyperband")
    elif plot == 'test':
        if marker:
            plt.plot(x[1:], y_test[1:], marker='.', label=r"Hyperband")
        else:
            plt.plot(x[1:], y_test[1:], label=r"Hyperband")

    os.chdir('..')

    # TPE
    os.chdir(paths[1])
    shortest = sys.maxsize

    valid_tracks = np.array([None for _ in range(runs_end - runs_begin)])
    test_tracks = np.array([None for _ in range(runs_end - runs_begin)])
    for i in range(runs_end-runs_begin):
        [trials, _] = cPickle.load(open(classifier_name + optimizer_name + str(i+runs_begin) + '/results.pkl', 'rb'))
        valid_track, _ = combine_tracks(trials)
        if len(valid_track) < shortest:
            shortest = len(valid_track)
    for i in range(runs_end-runs_begin):
        [trials, _] = cPickle.load(open(classifier_name + optimizer_name + str(i+runs_begin) + '/results.pkl', 'rb'))
        valid_track, test_track = combine_tracks(trials)
        valid_tracks[i] = valid_track[0:shortest]
        test_tracks[i] = test_track[0:shortest]

    # length = len(tracks[0])
    x = range(shortest)
    if resource_type == 'iterations':
        valid_tracks += 1
        test_tracks += 1
    y_valid = np.mean(valid_tracks, axis=0)
    y_test = np.mean(test_tracks, axis=0)
    if devs:
        if plot == 'valid':
            err = np.std(valid_tracks, axis=0) / 10
            lower = y_valid - err
            higher = y_valid + err
            plt.fill_between(x[1:], lower[1:], higher[1:], alpha=0.4)
    if plot == 'valid':
        if marker:
            plt.plot(x[1:], y_valid[1:], marker='*', label=r"TPE")
        else:
            plt.plot(x[1:], y_valid[1:], label=r"TPE")
    elif plot == 'test':
        if marker:
            plt.plot(x[1:], y_test[1:], marker='*', label=r"TPE")
        else:
            plt.plot(x[1:], y_test[1:], label=r"TPE")

    os.chdir('..')

    # GPO(POO)
    os.chdir(paths[2])
    shortest = sys.maxsize

    valid_losses = np.array([None for _ in range(runs_end-runs_begin)])
    test_losses = np.array([None for _ in range(runs_end-runs_begin)])
    for i in range(runs_end-runs_begin):
        valid_loss, test_loss = \
            cPickle.load(open(classifier_name + optimizer_name + str(i+runs_begin) + '/results.pkl', 'rb'))
        if len(valid_loss) < shortest:
            shortest = len(valid_loss)
    for i in range(runs_end-runs_begin):
        valid_loss, test_loss = \
            cPickle.load(open(classifier_name + optimizer_name + str(i+runs_begin) + '/results.pkl', 'rb'))
        valid_losses[i] = valid_loss[0:shortest]
        test_losses[i] = test_loss[0:shortest]

    # length = len(tracks[0])
    x = range(shortest)
    if resource_type == 'iterations':
        valid_losses = 1 - valid_losses
        test_losses += 1
    y_valid = np.mean(valid_losses, axis=0)
    y_test = np.mean(test_losses, axis=0)
    # if devs:
    #     if plot == 'valid':
    #         err = np.std(valid_tracks, axis=0) / 10
    #         lower = y_valid - err
    #         higher = y_valid + err
    #         plt.fill_between(x, lower, higher, alpha=0.4)
    # if plot == 'valid':
    #     if marker:
    #         plt.plot(x[1:], y_valid[1:], marker='+', label=r"GPO(HOO)")
    #     else:
    #         plt.plot(x[1:], y_valid[1:], label=r"GPO(HOO)")
    # elif plot == 'test':
    #     if marker:
    #         plt.plot(x[1:], y_test[1:], marker='+', label=r"GPO(HOO)")
    #     else:
    #         plt.plot(x[1:], y_test[1:], label=r"GPO(HOO)")

    os.chdir('..')

    # # PCT
    # os.chdir(paths[3])
    # shortest = sys.maxsize
    #
    # losses = np.array([None for _ in range(runs_end-runs_begin)])
    # for i in range(runs_end-runs_begin):
    #     loss = cPickle.load(open(classifier_name + optimizer_name + str(i+runs_begin) + '/results.pkl', 'rb'))
    #     if len(loss) < shortest:
    #         shortest = len(loss)
    # for i in range(runs_end-runs_begin):
    #     loss = cPickle.load(open(classifier_name + optimizer_name + str(i+runs_begin) + '/results.pkl', 'rb'))
    #     losses[i] = loss[0:shortest]
    #
    # # length = len(tracks[0])
    # x = range(shortest+1)
    # y = np.mean(losses, axis=0)
    # y = np.append([1.], y)
    # if devs:
    #     err = np.std(tracks, axis=0)
    #     lower = y - err
    #     higher = y + err
    #     plt.fill_between(x, lower, higher, facecolor='lightblue')
    # if type_plot == 'linear':
    #     if marker:
    #         plt.plot(x[1:], y[1:], label=r"GPO(PCT)")
    #     else:
    #         plt.plot(x[1:], y[1:], label=r"GPO(PCT)")
    # elif type_plot == 'log':
    #     plt.loglog(x[1:], y[1:], label=r"GPO(PCT)")
    #
    # os.chdir('..')

    # Random Search
    os.chdir(paths[3])
    shortest = sys.maxsize

    valid_tracks = np.array([None for _ in range(runs_end - runs_begin)])
    test_tracks = np.array([None for _ in range(runs_end - runs_begin)])
    for i in range(runs_end - runs_begin):
        [_, _, valid_track, _] = \
            cPickle.load(open(classifier_name + optimizer_name + str(i + runs_begin) + '/results.pkl', 'rb'))
        if len(valid_track) < shortest:
            shortest = len(valid_track)
    for i in range(runs_end - runs_begin):
        [_, _, valid_track, test_track] = \
            cPickle.load(open(classifier_name + optimizer_name + str(i + runs_begin) + '/results.pkl', 'rb'))
        valid_tracks[i] = valid_track[0:shortest]
        test_tracks[i] = test_track[0:shortest]

    # length = len(tracks[0])
    x = range(shortest)
    if resource_type == 'iterations':
        valid_tracks += 1
        test_tracks += 1
    y_valid = np.mean(valid_tracks, axis=0)
    y_test = np.mean(test_tracks, axis=0)
    if devs:
        if plot == 'valid':
            err = np.std(valid_tracks, axis=0) / 10
            lower = y_valid - err
            higher = y_valid + err
            plt.fill_between(x[1:], lower[1:], higher[1:], alpha=0.4)
    if plot == 'valid':
        if marker:
            plt.plot(x[1:], y_valid[1:], marker='x', label=r"Random Search")
        else:
            plt.plot(x[1:], y_valid[1:], label=r"Random Search")
    elif plot == 'test':
        if marker:
            plt.plot(x[1:], y_test[1:], marker='x', label=r"Random Search")
        else:
            plt.plot(x[1:], y_test[1:], label=r"Random Search")

    os.chdir('..')

    # Hyperloop
    os.chdir(paths[4])
    shortest = sys.maxsize

    valid_tracks = np.array([None for _ in range(runs_end - runs_begin)])
    test_tracks = np.array([None for _ in range(runs_end - runs_begin)])
    for i in range(runs_end - runs_begin):
        [_, _, valid_track, _] = \
            cPickle.load(open(classifier_name + optimizer_name + str(i + runs_begin) + '/results.pkl', 'rb'))
        if len(valid_track) < shortest:
            shortest = len(valid_track)
    for i in range(runs_end - runs_begin):
        [_, _, valid_track, test_track] = \
            cPickle.load(open(classifier_name + optimizer_name + str(i + runs_begin) + '/results.pkl', 'rb'))
        valid_tracks[i] = valid_track[0:shortest]
        test_tracks[i] = test_track[0:shortest]

    # length = len(tracks[0])
    x = range(shortest)
    if resource_type == 'iterations':
        valid_tracks += 1
        test_tracks += 1
    y_valid = np.mean(valid_tracks, axis=0)
    y_test = np.mean(test_tracks, axis=0)
    if devs:
        if plot == 'valid':
            err = np.std(valid_tracks, axis=0) / 10
            lower = y_valid - err
            higher = y_valid + err
            plt.fill_between(x[1:], lower[1:], higher[1:], alpha=0.4)
    if plot == 'valid':
        if marker:
            plt.plot(x[1:], y_valid[1:], marker='2', label=r"H-TTTS")
        else:
            plt.plot(x[1:], y_valid[1:], label=r"H-TTTS")
    elif plot == 'test':
        if marker:
            plt.plot(x[1:], y_test[1:], marker='2', label=r"H-TTTS")
        else:
            plt.plot(x[1:], y_test[1:], label=r"H-TTTS")

    os.chdir('..')

    # Dynamic TTTS
    os.chdir(paths[5])
    shortest = sys.maxsize

    valid_tracks = np.array([None for _ in range(runs_end - runs_begin)])
    test_tracks = np.array([None for _ in range(runs_end - runs_begin)])
    for i in range(runs_end - runs_begin):
        [_, _, valid_track, _] = \
            cPickle.load(open(classifier_name + optimizer_name + str(i + runs_begin) + '/results.pkl', 'rb'))
        if len(valid_track) < shortest:
            shortest = len(valid_track)
    for i in range(runs_end - runs_begin):
        [_, _, valid_track, test_track] = \
            cPickle.load(open(classifier_name + optimizer_name + str(i + runs_begin) + '/results.pkl', 'rb'))
        valid_tracks[i] = valid_track[0:shortest]
        test_tracks[i] = test_track[0:shortest]

    # length = len(tracks[0])
    x = range(shortest)
    if resource_type == 'iterations':
        valid_tracks += 1
        test_tracks += 1
    y_valid = np.mean(valid_tracks, axis=0)
    y_test = np.mean(test_tracks, axis=0)
    if devs:
        if plot == 'valid':
            err = np.std(valid_tracks, axis=0) / 10
            lower = y_valid - err
            higher = y_valid + err
            plt.fill_between(x[1:], lower[1:], higher[1:], alpha=0.4)
    if plot == 'valid':
        if marker:
            plt.plot(x[1:], y_valid[1:], marker='4', label=r"D-TTTS")
        else:
            plt.plot(x[1:], y_valid[1:], label=r"D-TTTS")
    elif plot == 'test':
        if marker:
            plt.plot(x[1:], y_test[1:], marker='4', label=r"D-TTTS")
        else:
            plt.plot(x[1:], y_test[1:], label=r"D-TTTS")

    os.chdir('..')

    # plt.grid()
    # plt.xlim((0, 400))
    # plt.ylim((0, 0.1))
    plt.legend(loc=0)
    if plot == 'valid':
        plt.ylabel('Validation error')
    else:
        plt.ylabel('Test error')
    if resource_type == 'epochs':
        plt.xlabel('Number of Epochs')
    elif resource_type == 'iterations':
        plt.xlabel('Number of Iterations')
    save_path = os.path.join(os.path.abspath('../'), 'img/{}'.format(classifier_name + str(idx)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(os.path.abspath('../'), 'img/{}/{}.pdf'.format(classifier_name + str(idx), dataset_name)))
    plt.close(fig)
