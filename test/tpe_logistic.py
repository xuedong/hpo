import time
import theano.tensor as ts

import src.classifiers.logistic as logistic
import src.utils as utils

from hyperopt import tpe
from hyperopt import fmin
from hyperopt import hp
from hyperopt import Trials
from hyperopt import STATUS_OK


EPOCHS = 1
DATA = utils.load_data('mnist.pkl.gz')


x = ts.matrix('x')
test_model = logistic.LogisticRegression(x, 28*28, 10)
params = test_model.get_search_space()
# arms = model.generate_arms(1, "../result/", params, True)
# train_loss, val_err, test_err = logistic.run_solver(1000, arms[0], data)

space = [
    hp.loguniform('learning_rate', 1 * 10 ** (-3), 1 * 10 ** (-1)),
    hp.randint('batch_size', 1000),
]
trials = Trials()


def solver(learning_rate, batch_size, epochs, data):
    arm = {'dir': ".", 'learning_rate': learning_rate, 'batch_size': batch_size, 'results': []}
    _, _, test_err, _ = test_model.run_solver(epochs, arm, data, verbose=True)

    return test_err


def objective(hps):
    learning_rate, batch_size = hps
    return {
        'loss': solver(learning_rate, batch_size, EPOCHS, DATA),
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time()
    }


if __name__ == "__main__":
    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=3,
                trials=trials)
    print(best)
    print(trials.trials)
    print(trials.results)
    print(trials.losses())
    print(trials.statuses())
