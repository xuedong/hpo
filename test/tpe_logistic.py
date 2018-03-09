import time
import theano.tensor as ts

import src.classifiers.logistic as logistic
import src.utils as utils
import src.bo.tpe_hyperopt as tpe_hyperopt

from hyperopt import tpe
from hyperopt import fmin
# from hyperopt import hp
from hyperopt import Trials
from hyperopt import STATUS_OK


EPOCHS = 1
DATA = utils.load_data('mnist.pkl.gz')


x = ts.matrix('x')
test_model = logistic.LogisticRegression(x, 28*28, 10)
params = test_model.get_search_space()

trials = Trials()


def objective(hps):
    learning_rate, batch_size = hps
    arm = {'dir': ".", 'learning_rate': learning_rate, 'batch_size': int(batch_size), 'results': []}
    return {
        'loss': tpe_hyperopt.solver(test_model, EPOCHS, arm, DATA, verbose=True),
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time()
    }


if __name__ == "__main__":
    best = fmin(objective,
                space=tpe_hyperopt.convert_params(params),
                algo=tpe.suggest,
                max_evals=3,
                trials=trials)
    print(best)
    print(trials.trials)
    print(trials.results)
    print(trials.losses())
    print(trials.statuses())
