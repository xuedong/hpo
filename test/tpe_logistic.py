import timeit
import theano.tensor as ts
from six.moves import cPickle

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
    start_time = timeit.default_timer()
    learning_rate, batch_size = hps
    arm = {'dir': "../result/tpe_logistic_0", 'learning_rate': learning_rate, 'batch_size': int(batch_size), 'results': []}
    train_loss, best_valid_loss, test_score, track = test_model.run_solver(EPOCHS, arm, DATA, verbose=True)
    return {
        'loss': test_score,
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': timeit.default_timer() - start_time,
        # -- attachments are handled differently
        'attachments':
            {'track': cPickle.dumps(track)}
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
    msg = trials.trial_attachments(trials.trials[0])['track']
    current_track = cPickle.loads(msg)
    print(current_track)
