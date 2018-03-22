import math
import numpy as np
import timeit
import theano.tensor as ts

import src.target as target
# import src.ho.hoo as hoo
# import src.ho.hct as hct
import src.ho.utils_ho as utils_ho
import src.utils as utils
import src.classifiers.logistic as logistic


if __name__ == '__main__':
    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)

    start = timeit.default_timer()

    x = ts.matrix('x')
    test_model = logistic.LogisticRegression(x, 28*28, 10)
    params = test_model.get_search_space()

    f_target = target.TheanoLogistic(10, data)
    bbox = utils_ho.std_box(f_target.f, None, 2, 0.1,
                            [(params['learning_rate'].get_min(), params['learning_rate'].get_max()),
                             (params['batch_size'].get_min(), params['batch_size'].get_max())],
                            [params['learning_rate'].get_type(), params['batch_size'].get_type()])

    current = [0. for _ in range(100)]
    c = 2 * math.sqrt(1./(1-0.66))
    c1 = (0.66/(3*1.)) ** (1./8)
    losses = utils_ho.loss_hct(bbox=bbox, rho=0.66, nu=1., c=c, c1=c1, delta=0.05, horizon=100)
