import math
import numpy as np
import timeit
import theano.tensor as ts
import matplotlib.pyplot as plt

import source.target as target
# import source.ho.hoo as hoo
# import source.ho.hct as hct
import source.ho.utils_ho as utils_ho
import source.utils as utils
import source.classifiers.logistic as logistic


if __name__ == '__main__':
    horizon = 1000
    c = 2 * math.sqrt(1. / (1 - 0.66))
    c1 = (0.66 / (3 * 1.)) ** (1. / 8)
    # f_target = target.Rosenbrock(1, 100)
    # bbox = utils_ho.std_box(f_target.f, f_target.fmax, 2, 0.1, [(-3., 3.), (-3., 3.)], ['cont', 'cont'])
    #
    # regrets = np.zeros(horizon)
    # for k in range(10):
    #     current, _, _ = utils_ho.regret_hct(bbox=bbox, rho=0.66, nu=1., c=c, c1=c1, delta=0.05, horizon=horizon)
    #     regrets = np.add(regrets, current)
    #     # print(current)
    # x = range(horizon)
    # plt.plot(x, regrets/10.)
    # plt.show()

    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)

    start = timeit.default_timer()

    x = ts.matrix('x')
    test_model = logistic.LogisticRegression(x, 28*28, 10)
    params = test_model.get_search_space()

    f_target = target.TheanoLogistic(1, data, ".")
    bbox = utils_ho.std_box(f_target.f, None, 2, 0.1,
                            [(params['learning_rate'].get_min(), params['learning_rate'].get_max()),
                             (params['batch_size'].get_min(), params['batch_size'].get_max())],
                            [params['learning_rate'].get_type(), params['batch_size'].get_type()])

    current = [0. for _ in range(horizon)]
    alpha = math.log(10) * (0.1 ** 2)
    losses = utils_ho.loss_hct(bbox=bbox, rho=0.66, nu=1., c=c, c1=c1, delta=0.05, horizon=horizon)
    # losses = utils_ho.loss_hoo(bbox=bbox, rho=0.66, nu=1., alpha=alpha, sigma=0.1, horizon=10, update=False)
    print(losses)
