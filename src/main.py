import numpy as np
import sys
import os
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

import classifiers.logistic as logistic
import classifiers.mlp as mlp
import hyperband.hyperband_finite as hyperband_finite
import bo.tpe_hyperopt as tpe_hyperopt


def main(model):
    data_dir = 'mnist.pkl.gz'
    data = utils.load_data(data_dir)

    output_dir = ''
    # rng = np.random.RandomState(12345)
    random.seed(12345)
    model_name = model + '_sgd_'
    exp_name = 'tpe_' + model + '_0/'

    for seed_id in range(2):
        director = output_dir + '../result/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(director):
            os.makedirs(director)
        log_dir = output_dir + '../log/' + exp_name + model_name + str(seed_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sys.stdout = logger.Logger(log_dir, 'tpe')

        x = ts.matrix('x')
        test_model = logistic.LogisticRegression(x, 28*28, 10)
        params = test_model.get_search_space()

        # <-- Running Hyperband

        # start_time = timeit.default_timer()

        # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4,
        #                                   verbose=True)
        # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=0,
        #                                   verbose=True)
        # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=1,
        #                                   verbose=True)
        # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=2,
        #                                   verbose=True)
        # hyperband_finite.hyperband_finite(test_model, 'epoch', params, 1, 100, 360, director, data, eta=4, s_run=3,
        #                                   verbose=True)
        # hyperband_finite.hyperband_finite(model, 'epoch', params, 1, 81, 360, director, data, eta=4, s_run=4)
        # print(train_loss, val_acc, test_acc)

        # end_time = timeit.default_timer()

        # <-- Running TPE

        start_time = timeit.default_timer()

        trials = Trials()

        def objective(hps):
            learning_rate, batch_size = hps
            arm = {'dir': director,
                   'learning_rate': learning_rate, 'batch_size': int(batch_size), 'results': []}
            train_loss, best_valid_loss, test_score, track = test_model.run_solver(1, arm, data, verbose=True)
            return {
                'loss': test_score,
                'status': STATUS_OK,
                # -- store other results like this
                'train_loss': train_loss,
                'valid_loss': best_valid_loss,
                # -- attachments are handled differently
                'attachments':
                    {'track': cPickle.dumps(track)}
            }

        best = fmin(objective,
                    space=tpe_hyperopt.convert_params(params),
                    algo=tpe.suggest,
                    max_evals=4,
                    trials=trials)

        with open(director + '/result.pkl', 'wb') as file:
            cPickle.dump([trials, best], file)

        end_time = timeit.default_timer()

        print(('The code for the trial number ' +
               str(seed_id) +
               ' ran for %.1fs' % (end_time - start_time)), file=sys.stderr)


if __name__ == "__main__":
    main('logistic')
