#!/usr/bin/env python

# This code is based on the original code of Jean-Bastien Grill
# https://team.inria.fr/sequel/Software/POO/

# import pylab as pl
import numpy as np
import progressbar
import math
# import random
# import pickle
from six.moves import cPickle

import ho.hoo as hoo
import ho.poo as poo
import ho.hct as hct


def std_box(target, fmax, nsplits, sigma, support, support_type):
    box = Box(target, fmax, nsplits, sigma, support, support_type)
    box.std_partition()
    box.std_noise(sigma)

    return box


class Box:
    def __init__(self, target, fmax, nsplits, sigma, support, support_type):
        self.target = target
        self.f_noised = None
        self.f_mean = target.f
        self.fmax = fmax
        self.split = None
        self.center = None
        self.rand = None
        self.nsplits = nsplits
        self.sigma = sigma
        self.support = support
        self.support_type = support_type

    def std_partition(self):
        """Standard partitioning of a black box.
        """
        self.center = std_center
        self.rand = std_rand
        self.split = std_split_rand

    def std_noise(self, sigma):
        """Stochastic target with Gaussian or uniform noise.
        """
        self.f_noised = lambda x: self.f_mean(x) + sigma * np.random.normal(0, sigma)
        # self.f_noised = lambda x: self.f_mean(x) + sigma*random.random()

    """
    def plot1D(self):
        a, b = self.side
        fig, ax = plt.subplots()
        ax.set_xlim(a, b)
        x = np.array([a+i/10000. for i in range(int((b-a)*10000))])
        y = np.array([self.f_mean([x[i]]) for i in range(int((b-a)*10000))])
        plt.plot(x, y)
        plt.show()

    def plot2D(self):
        # 2D spaced down level curve plot
        x = np.array([(i-600)/100. for i in range(1199)])
        y = np.array([(j-600)/100. for j in range(1199)])
        x, Y = pl.meshgrid(x, y)
        Z = np.array([[self.f_mean([(i-600)/100., (j-600)/100.]) for i in range(1199)] for j in range(1199)])

        im = pl.imshow(Z, cmap=pl.cm.RdBu)
        cset = pl.contour(Z, np.arange(-1, 1.5, 0.2), linewidths=2, cmap=pl.cm.Set2)
        pl.colorbar(im)
        pl.show()

        # 3D plot
        fig = mpl.pyplot.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, Y, Z, rstride=1, cstride=1, cmap=pl.cm.RdBu, linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(mpl.ticker.LinearLocator(10))
        ax.zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        mpl.pyplot.show()
    """


# Regret for HOO
def regret_hoo(bbox, rho, nu, alpha, sigma, horizon, update):
    # y = np.zeros(horizon)
    y_cum = [0. for _ in range(horizon)]
    y_sim = [0. for _ in range(horizon)]
    x_sel = [None for _ in range(horizon)]
    htree = hoo.HTree(bbox.support, bbox.support_type, None, 0, rho, nu, sigma, bbox)
    cum = 0.

    for i in range(horizon):
        if update and alpha < math.log(i + 1) * (sigma ** 2):
            alpha += 1
            htree.update(alpha)
        x, _, _ = htree.sample(alpha)
        cum += bbox.fmax - bbox.f_mean(x)
        y_cum[i] = cum / (i + 1)
        x_sel[i] = x
        z = x_sel[0:(i+1)][np.random.choice(len(x_sel[0:(i+1)]))]
        y_sim[i] = bbox.fmax - bbox.f_mean(z)

    return y_cum, y_sim, x_sel


def loss_hoo(bbox, rho, nu, alpha, sigma, horizon, update, director, keep=False):
    losses = [0. for _ in range(horizon)]
    htree = hoo.HTree(bbox.support, bbox.support_type, None, 0, rho, nu, sigma, bbox)
    best = 1.
    test = 1.

    bar = progressbar.ProgressBar()

    track_valid = np.array([1.])
    track_test = np.array([1.])
    for i in bar(range(horizon)):
        # print(str(i+1) + '/' + str(horizon))
        if update and alpha < math.log(i + 1) * (sigma ** 2):
            alpha += 1
            htree.update(alpha)
        x, current, _, _ = htree.sample(alpha)
        if not keep:
            [_, test_score] = cPickle.load(open(director + '/tracks.pkl', 'rb'))
            if -current < best:
                best = -current
                test = test_score
                losses[i] = test
            else:
                losses[i] = test
        else:
            [current_track_valid, current_track_test] = cPickle.load(open(director + '/tracks.pkl', 'rb'))
            # print(current_track_test)
            current_best_valid = track_valid[-1]
            current_test = track_test[-1]
            for j in range(1, len(current_track_valid)):
                if current_track_valid[j] < current_best_valid:
                    current_best_valid = current_track_valid[j]
                    current_test = current_track_test[j]
                    track_valid = np.append(track_valid, current_best_valid)
                    track_test = np.append(track_test, current_test)
                else:
                    track_valid = np.append(track_valid, current_best_valid)
                    track_test = np.append(track_test, current_test)

    if keep:
        losses = track_test[1:]

    return losses


def regret_hct(bbox, rho, nu, c, c1, delta, sigma, horizon):
    y_cum = [0. for _ in range(horizon)]
    y_sim = [0. for _ in range(horizon)]
    x_sel = [None for _ in range(horizon)]
    hctree = hct.HCTree(bbox.support, bbox.support_type, None, 0, rho, nu, 1, 1, sigma, bbox)
    cum = 0.

    for i in range(1, horizon+1):
        tplus = int(2 ** (math.ceil(math.log(i))))
        dvalue = min(c1 * delta/tplus, 0.5)

        if i == tplus:
            hctree.update(c, dvalue)
        x, _, _ = hctree.sample(c, dvalue)
        cum += bbox.fmax - bbox.f_mean(x)
        y_cum[i-1] = cum/i
        x_sel[i-1] = x
        z = x_sel[0:i][np.random.choice(len(x_sel[0:i]))]
        y_sim[i-1] = bbox.fmax - bbox.f_mean(z)

    return y_cum, y_sim, x_sel


def loss_hct(bbox: Box, rho, nu, c, c1, delta, sigma, horizon, director, keep=False):
    losses = [0. for _ in range(horizon)]
    hctree = hct.HCTree(bbox.support, bbox.support_type, None, 0, rho, nu, 1, 1, sigma, bbox)
    best = 1.
    test = 1.

    bar = progressbar.ProgressBar()

    for i in bar(range(horizon)):
        tplus = int(2 ** (math.ceil(math.log(i+1))))
        dvalue = min(c1 * delta / tplus, 0.5)

        if i+1 == tplus:
            hctree.update(c, dvalue)
        x, current, _, _ = hctree.sample(c, dvalue)
        if keep:
            if hctree.get_change_status():
                bbox.target.set_status(True)
            else:
                bbox.target.set_status(False)
            hctree.reset_change_status()
            [best_valid_loss, test_score] = cPickle.load(open(director + '/tracks.pkl', 'rb'))
            if best_valid_loss < best:
                best = best_valid_loss
                test = test_score
                losses[i] = test
            else:
                losses[i] = test
        # current = bbox.f_mean(x)
        else:
            [_, test_score] = cPickle.load(open(director + '/tracks.pkl', 'rb'))
            if -current < best:
                best = -current
                test = test_score
                losses[i] = test
            else:
                losses[i] = test

    return losses


def regret_poo(bbox, rhos, nu, alpha, horizon, epoch):
    y_cum = [[0. for _ in range(horizon)] for _ in range(epoch)]
    y_sim = [[0. for _ in range(horizon)] for _ in range(epoch)]
    x_sel = [[None for _ in range(horizon)] for _ in range(epoch)]
    for i in range(epoch):
        ptree = poo.PTree(bbox.support, bbox.support_type, None, 0, rhos, nu, bbox)
        count = 0
        length = len(rhos)
        cum = [0.] * length
        emp = [0.] * length
        smp = [0] * length

        while count < horizon:
            for k in range(length):
                x, noisy, existed = ptree.sample(alpha, k)
                cum[k] += bbox.fmax - bbox.f_mean(x)
                count += existed
                emp[k] += noisy
                smp[k] += 1

                if existed and count <= horizon:
                    best_k = max(range(length), key=lambda a: (-float("inf") if smp[k] == 0 else emp[a] / smp[k]))
                    y_cum[i][count - 1] = 1 if smp[best_k] == 0 else cum[best_k] / float(smp[best_k])
                    x_sel[i][count - 1] = x
                    z = x_sel[i][0:count][np.random.choice(len(x_sel[i][0:count]))]
                    y_sim[i][count - 1] = bbox.fmax - bbox.f_mean(z)

    return y_cum, y_sim, x_sel


def loss_poo(bbox, rhos, nu, alpha, horizon, epoch):
    losses = [[0. for _ in range(horizon)] for _ in range(epoch)]
    for i in range(epoch):
        ptree = poo.PTree(bbox.support, bbox.support_type, None, 0, rhos, nu, bbox)
        count = 0
        length = len(rhos)
        cum = [0.] * length
        emp = [0.] * length
        smp = [0] * length

        while count < horizon:
            for k in range(length):
                x, noisy, existed = ptree.sample(alpha, k)
                cum[k] += bbox.f_mean(x)
                count += existed
                emp[k] += noisy
                smp[k] += 1

                if existed and count <= horizon:
                    best_k = max(range(length), key=lambda a: (-float("inf") if smp[k] == 0 else emp[a] / smp[k]))
                    losses[i][count - 1] = -1 if smp[best_k] == 0 else cum[best_k] / float(smp[best_k])

    return losses


# Plot regret curve
# def show(path, epoch, horizon, rhos_hoo, rhos_poo, delta):
#     data = [None for _ in range(epoch)]
#     for k in range(epoch):
#         with open(path + "HOO" + str(k + 1), 'rb') as file:
#             data[k] = pickle.load(file)
#     with open(path + "POO", 'rb') as file:
#         data_poo = pickle.load(file)
#     # with open(path+"GPUCB_DIRECT", 'rb') as file:
#     #     data_ucb_direct = pickle.load(file)
#     # with open(path+"GPUCB_LBFGS", 'rb') as file:
#     #     data_ucb_lbfgs = pickle.load(file)
#     # with open(path+"EI", 'rb') as file:
#     #     data_ei = pickle.load(file)
#     # with open(path+"PI", 'rb') as file:
#     #     data_pi = pickle.load(file)
#     # with open(path+"THOMP", 'rb') as file:
#     #     data_thomp = pickle.load(file)
#
#     length_hoo = len(rhos_hoo)
#     length_poo = len(rhos_poo)
#     rhostoshow = [int(length_hoo * k / 4.) for k in range(4)]
#     # rhostoshow = [0, 6, 12, 18]
#     # rhostoshow = [0, 1, 3, 7, 15]
#     style = [[5, 5], [1, 3], [5, 3, 1, 3], [5, 2, 5, 2, 5, 10]]
#     # style = [[5, 5], [1, 3], [5, 3, 1, 3], [5, 2, 5, 2, 5, 10], [3, 1]]
#
#     means = [[sum([data[k][j][i] for k in range(epoch)]) / float(epoch)
#               for i in range(horizon)] for j in range(length_hoo)]
#     devs = [
#         [math.sqrt(sum([(data[k][j][i] - means[j][i]) ** 2
#                         for k in range(epoch)]) / (float(epoch) * float(epoch - 1)))
#          for i in range(horizon)] for j in range(length_hoo)]
#
#     means_poo = [sum([data_poo[u][v] / float(epoch) for u in range(epoch)]) for v in range(horizon)]
#
#     # means_ucb_direct = [sum([data_ucb_direct[u][v]/float(epoch) for u in range(epoch)]) for v in range(horizon)]
#
#     # means_ucb_lbfgs = [sum([data_ucb_lbfgs[u][v]/float(epoch) for u in range(epoch)]) for v in range(horizon)]
#
#     # means_ei = [sum([data_ei[u][v]/float(epoch) for u in range(epoch)]) for v in range(horizon)]
#
#     # means_pi = [sum([data_pi[u][v]/float(epoch) for u in range(epoch)]) for v in range(horizon)]
#
#     # means_thomp = [sum([data_thomp[u][v]/float(epoch) for u in range(epoch)]) for v in range(horizon)]
#
#     x = np.array(range(horizon))
#     for i in range(len(rhostoshow)):
#         k = rhostoshow[i]
#         label__ = r"$\mathtt{HOO}, \rho = " + str(float(k) / float(length_hoo)) + "$"
#         pl.plot(x, np.array(means[k]), label=label__, dashes=style[i])
#     pl.plot(x, np.array(means_poo), label=r"$\mathtt{POO}$")
#     # pl.plot(x, np.array(means_ucb_direct), label=r"$\mathtt{GPUCB-DIRECT}$", color='blue')
#     # pl.plot(x, np.array(means_ucb_lbfgs), label=r"$\mathtt{GPUCB-LBFGS}$", color='green')
#     # pl.plot(x, np.array(means_ei), label=r"$\mathtt{EI}$", color='cyan')
#     # pl.plot(x, np.array(means_pi), label=r"$\mathtt{PI}$", color='magenta')
#     # pl.plot(x, np.array(means_thomp), label=r"$\mathtt{THOMPSON}$", color='magenta')
#     pl.legend()
#     pl.xlabel("number of evaluations")
#     pl.ylabel("simple regret")
#     pl.show()
#
#     x = np.array(map(math.log, range(horizon)[1:]))
#     for i in range(len(rhostoshow)):
#         k = rhostoshow[i]
#         label__ = r"$\mathtt{HOO}, \rho = " + str(float(k) / float(length_hoo)) + "$"
#         pl.plot(x, np.array(map(math.log, means[k][1:])), label=label__, dashes=style[i])
#     pl.plot(x, np.array(map(math.log, means_poo[1:])), label=r"$\mathtt{POO}$")
#     # pl.plot(x, np.array(map(math.log, means_ucb_direct[1:])), label=r"$\mathtt{GPUCB-DIRECT}$", color='blue')
#     # pl.plot(x, np.array(map(math.log, means_ucb_lbfgs[1:])), label=r"$\mathtt{GPUCB-LBFGS}$", color='green')
#     # pl.plot(x, np.array(map(math.log, means_ei[1:])), label=r"$\mathtt{EI}$", color='cyan')
#     # pl.plot(x, np.array(map(math.log, means_pi[1:])), label=r"$\mathtt{PI}$", color='magenta')
#     # pl.plot(x, np.array(map(math.log, means_thomp[1:])), label=r"$\mathtt{THOMPSON}$", color='magenta')
#     pl.legend(loc=3)
#     pl.xlabel("number of evaluations (log-scale)")
#     pl.ylabel("simple regret")
#     pl.show()
#
#     x = np.array([float(j) / float(length_hoo - 1) for j in range(length_hoo)])
#     y = np.array([means[j][horizon - 1] for j in range(length_hoo)])
#     z1 = np.array([means[j][horizon - 1] + math.sqrt(2 * (devs[j][horizon - 1] ** 2) * math.log(1 / delta)) for j in
#                    range(length_hoo)])
#     # z1 = np.array([means[j][horizon-1]+2*devs[j][horizon-1] for j in range(length_hoo)])
#     z2 = np.array([means[j][horizon - 1] - math.sqrt(2 * (devs[j][horizon - 1] ** 2) * math.log(1 / delta)) for j in
#                    range(length_hoo)])
#     # z2 = np.array([means[j][horizon-1]-2*devs[j][horizon-1] for j in range(length_hoo)])
#     pl.plot(x, y)
#     pl.plot(x, z1, color="green")
#     pl.plot(x, z2, color="green")
#     pl.xlabel(r"$\rho$")
#     pl.ylabel("simple regret after " + str(horizon) + " evaluations")
#     pl.show()
#
#     x = np.array([float(j) / float(length_hoo - 1) for j in range(length_hoo)])
#     y = np.array([means[j][horizon - 1] for j in range(length_hoo)])
#     err = np.array([math.sqrt(2 * (devs[j][horizon - 1] ** 2) * math.log(1 / delta)) for j in range(length_hoo)])
#     # err = np.array([2*devs[j][horizon-1] for j in range(length_hoo)])
#     pl.errorbar(x, y, yerr=err, color='black', errorevery=3)
#     pl.xlabel(r"$\rho$")
#     pl.ylabel("simple regret after " + str(horizon) + " evaluations")
#     pl.show()


# How to choose rhos to be used
def get_rhos(nsplits, rhomax, horizon):
    dmax = math.log(nsplits) / math.log(1. / rhomax)
    n = 0
    big_n = 1
    rhos = np.array([rhomax])

    while n < horizon:
        if n == 0 or n == 1:
            threshold = -float("inf")
        else:
            threshold = 0.5 * dmax * math.log(n / math.log(n))

        while big_n <= threshold:
            for i in range(big_n):
                rho_current = math.pow(rhomax, 2. * big_n / (2 * (i + 1)))
                rhos = np.append(rhos, rho_current)
            n = 2 * n
            big_n = 2 * big_n

        n = n + big_n

    return np.unique(np.sort(rhos))


# Function domain partitioning
def std_center(support, support_type):
    """Pick the center of a subregion.
    """
    centers = []
    for i in range(len(support)):
        if support_type[i] == 'int' or 'integer':
            a, b = support[i]
            center = (a+b)/2
            centers.append(center)
        elif support_type[i] == 'cont' or 'continuous':
            a, b = support[i]
            center = (a+b)/2.
            centers.append(center)
        else:
            raise ValueError('Unsupported variable type.')

    return centers


def std_rand(support, support_type):
    """Randomly pick a point in a subregion.
    """
    rands = []
    for i in range(len(support)):
        if support_type[i] == 'int' or 'integer':
            a, b = support[i]
            rand = np.random.randint(a, b+1)
            rands.append(rand)
        elif support_type[i] == 'cont' or 'continuous':
            a, b = support[i]
            rand = a + (b-a)*np.random.random()
            rands.append(rand)
        else:
            raise ValueError('Unsupported variable type.')

    return rands


def std_split(support, support_type, nsplits):
    """Split a box uniformly.

    :param support: vector of support in each dimension
    :param support_type: continuous or discrete
    :param nsplits: number of splits
    :return:
    """
    lens = np.array([support[i][1]-support[i][0] for i in range(len(support))])
    max_index = np.argmax(lens)
    max_length = np.max(lens)
    a, b = support[max_index]
    step = max_length/float(nsplits)
    if support_type[max_index] == 'int' or 'integer':
        split = [(a+int(step*i), a+int(step*(i+1))) for i in range(nsplits)]
    elif support_type[max_index] == 'cont' or 'continuous':
        split = [(a+step*i, a+step*(i+1)) for i in range(nsplits)]
    else:
        raise ValueError("Unsupported variable type.")

    supports = [None for _ in range(nsplits)]
    supports_type = [None for _ in range(nsplits)]
    for i in range(nsplits):
        supports[i] = [support[j] for j in range(len(support))]
        supports[i][max_index] = split[i]
        supports_type[i] = support_type

    return supports, supports_type


def std_split_rand(support, support_type, nsplits):
    """Split a box randomly.

    :param support:
    :param support_type:
    :param nsplits:
    :return:
    """
    # lens = np.array([support[i][1] - support[i][0] for i in range(len(support))])
    rand_index = np.random.choice(len(support))
    rand_length = support[rand_index][1] - support[rand_index][0]
    a, b = support[rand_index]
    step = rand_length / float(nsplits)
    if support_type[rand_index] == 'int' or 'integer':
        split = [(a + int(step * i), a + int(step * (i + 1))) for i in range(nsplits)]
    elif support_type[rand_index] == 'cont' or 'continuous':
        split = [(a + step * i, a + step * (i + 1)) for i in range(nsplits)]
    else:
        raise ValueError("Unsupported variable type.")

    supports = [None for _ in range(nsplits)]
    supports_type = [None for _ in range(nsplits)]
    for i in range(nsplits):
        supports[i] = [support[j] for j in range(len(support))]
        supports[i][rand_index] = split[i]
        supports_type[i] = support_type

    return supports, supports_type
