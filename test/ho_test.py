import math
import numpy as np
import timeit
import matplotlib.pyplot as plt

import source.utils as utils
import source.target as target
import source.ho.utils_ho as utils_ho
import source.classifiers.logistic as logistic


if __name__ == '__main__':
    horizon = 10
    mcmc = 1
    rho = 0.66
    nu = 1.
    sigma = 0.1
    delta = 0.05
    alpha = math.log(horizon) * (sigma ** 2)
    c = 2 * math.sqrt(1. / (1 - 0.66))
    c1 = (0.66 / (3 * 1.)) ** (1. / 8)
    f_target = target.Rosenbrock(1, 100)
    bbox = utils_ho.std_box(f_target.f, f_target.fmax, 2, 0.1, [(-3., 3.), (-3., 3.)], ['cont', 'cont'])

    regrets = np.zeros(horizon)
    for k in range(10):
        current, _, _ = utils_ho.regret_hct(bbox=bbox, rho=0.66, nu=1., c=c, c1=c1, delta=0.05, sigma=sigma,
                                            horizon=horizon)
        regrets = np.add(regrets, current)
        # print(current)
    x = range(horizon)
    plt.plot(x, regrets/10.)
    plt.show()

    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)

    start = timeit.default_timer()

    # x = ts.matrix('x')
    # test_model = logistic.LogisticRegression(x, 28*28, 10)
    params = logistic.LogisticRegression.get_search_space()

    f_target = target.TheanoHOOLogistic(1, data, ".")
    bbox = utils_ho.std_box(f_target, None, 2, 0.1,
                            [(params['learning_rate'].get_min(), params['learning_rate'].get_max()),
                             (params['batch_size'].get_min(), params['batch_size'].get_max())],
                            [params['learning_rate'].get_type(), params['batch_size'].get_type()])

    current = [0. for _ in range(horizon)]
    alpha = math.log(10) * (0.1 ** 2)
    losses = utils_ho.loss_hct(bbox=bbox, rho=0.66, nu=1., c=c, c1=c1, delta=0.05, sigma=0.1, director='.',
                               horizon=horizon)
    # losses = utils_ho.loss_hoo(bbox=bbox, rho=0.66, nu=1., alpha=alpha, sigma=0.1, horizon=10, update=False)
    print(losses)
