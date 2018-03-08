import numpy as np
import sys
import os
# import random
import timeit
import theano.tensor as ts

import logger
import utils

# import logistic
import classifiers.mlp as mlp
import hyperband.hyperband_finite as hyperband_finite


def main(model):
    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)

    output_dir = ''
    rng = np.random.RandomState(12345)
    # random.seed(12345)
    model_name = model + '_sgd_'
    exp_name = 'hyperband_' + model + '_0/'

    for seed_id in range(5):
        start_time = timeit.default_timer()

        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir)

        x = ts.matrix('x')
        test_model = mlp.MLP(x, 28*28, 500, 10, rng=rng)
        params = test_model.get_search_space()
        # arms = model.generate_arms(1, "../result/", params, True)
        # train_loss, val_err, test_err = logistic.run_solver(1000, arms[0], data)

        hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4,
                                          verbose=True)
        hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=0,
                                          verbose=True)
        hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=1,
                                          verbose=True)
        hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=2,
                                          verbose=True)
        hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=3,
                                          verbose=True)
        # hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=4, s_run=4)
        # print(train_loss, val_acc, test_acc)

        end_time = timeit.default_timer()
        print(('The code for the trial number ' +
               str(seed_id) +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)


if __name__ == "__main__":
    main('mlp')
