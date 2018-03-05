import sys
import os
import random
import timeit
import theano.tensor as ts

import logger
import utils
import logistic
import hyperband_finite


def main():
    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)

    output_dir = ''
    random.seed(12345)
    model_name = 'logistic_sgd_'
    exp_name = 'hyperband_logistic_2/'

    for seed_id in range(10):
        start_time = timeit.default_timer()

        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir)

        x = ts.matrix('x')
        model = logistic.LogisticRegression(x, 28*28, 10)
        params = logistic.get_search_space()
        # arms = model.generate_arms(1, "../result/", params, True)
        # train_loss, val_err, test_err = logistic.run_solver(1000, arms[0], data)

        # hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 100, 360, director, data, eta=4)
        hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 4, 360, director, data, eta=4, s_run=0)
        # hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=1)
        # hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=2)
        # hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=3)
        # hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=4, s_run=4)
        # print(train_loss, val_acc, test_acc)

        end_time = timeit.default_timer()
        print(('The code for the trial number ' +
               str(seed_id) +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)


if __name__ == "__main__":
    main()
