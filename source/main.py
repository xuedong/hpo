import numpy as np
import sys
import os
import math
import random
import timeit
import theano.tensor as ts
from six.moves import cPickle

import log.logger as logger
import utils

from hyperopt import tpe
from hyperopt import fmin
from hyperopt import Trials
from hyperopt import STATUS_OK

import target
import classifiers.logistic as logistic
import classifiers.mlp as mlp
import hyperband.hyperband_finite as hyperband_finite
import bo.tpe_hyperopt as tpe_hyperopt
import ho.utils_ho as utils_ho


def main(model):
    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)

    output_dir = ''
    # rng = np.random.RandomState(12345)
    random.seed(12345)
    model_name = model + '_sgd_'
    exp_name = 'hyperband_' + model + '_3/'

    for seed_id in range(1):
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'hyperband')

        x = ts.matrix('x')
        test_model = logistic.LogisticRegression(x, 28*28, 10)
        # test_model = mlp.MLP(x, 28*28, 500, 10, rng)
        params = logistic.LogisticRegression.get_search_space()
        # params = mlp.MLP.get_search_space()

        # <-- Running Hyperband

        start_time = timeit.default_timer()

        hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 9, 360, director, data, eta=4,
                                          verbose=True)
        # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 1000, 360, director, data, eta=4, s_run=0,
        #                                   verbose=False)
        # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=1,
        #                                   verbose=True)
        # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=2,
        #                                   verbose=True)
        # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=3,
        #                                   verbose=True)
        # hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=4, s_run=4)

        end_time = timeit.default_timer()

        # <-- Running TPE

        # start_time = timeit.default_timer()
        #
        # trials = Trials()
        #
        # def objective(hps):
        #     learning_rate, batch_size, l2_reg = hps
        #     arm = {'dir': director,
        #            'learning_rate': learning_rate, 'batch_size': int(batch_size), 'n_hidden': 500,
        #            'l1_reg': 0., 'l2_reg': l2_reg,
        #            'results': []}
        #     train_loss, best_valid_loss, test_score, track = test_model.run_solver(100, arm, data, verbose=True)
        #     return {
        #         'loss': test_score,
        #         'status': STATUS_OK,
        #         # -- store other results like this
        #         'train_loss': train_loss,
        #         'valid_loss': best_valid_loss,
        #         # -- attachments are handled differently
        #         'attachments':
        #             {'track': cPickle.dumps(track)}
        #     }
        #
        # best = fmin(objective,
        #             space=tpe_hyperopt.convert_params(params),
        #             algo=tpe.suggest,
        #             max_evals=4,
        #             trials=trials)
        #
        # with open(director + '/results.pkl', 'wb') as file:
        #     cPickle.dump([trials, best], file)
        #
        # end_time = timeit.default_timer()

        # <-- Running HOO

        # start_time = timeit.default_timer()
        #
        # f_target = target.TheanoLogistic(1, data, director)
        # bbox = utils_ho.std_box(f_target.f, None, 2, 0.1,
        #                         [(params['learning_rate'].get_min(), params['learning_rate'].get_max()),
        #                          (params['batch_size'].get_min(), params['batch_size'].get_max())],
        #                         [params['learning_rate'].get_type(), params['batch_size'].get_type()])
        #
        # alpha = math.log(10) * (0.1 ** 2)
        # # losses = utils_ho.loss_hct(bbox=bbox, rho=0.66, nu=1., c=c, c1=c1, delta=0.05, horizon=10)
        # losses = utils_ho.loss_hoo(bbox=bbox, rho=0.66, nu=1., alpha=alpha, sigma=0.1, horizon=401, update=False)
        # losses = np.array(losses)
        #
        # with open(director + '/results.pkl', 'wb') as file:
        #     cPickle.dump(-losses, file)
        #
        # end_time = timeit.default_timer()

        print(('The code for the trial number ' +
               str(seed_id) +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)


if __name__ == "__main__":
    main('logistic')
    # main('mlp')
