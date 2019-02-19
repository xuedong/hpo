import numpy as np
from six.moves import cPickle

from hyperopt import hp


def convert_params(params):
    """

    :param params: hyperparamter dictionary as defined in params.py
    :return: search space for hyperopt
    """
    space = []
    for param in params:
        if params[param].get_type() == 'integer':
            space.append(hp.quniform(params[param].get_name(),
                                     params[param].get_min(), params[param].get_max(), params[param].interval))
        elif params[param].get_type() == 'continuous':
            if params[param].get_scale() == 'linear' and params[param].get_dist() == 'uniform':
                space.append(hp.uniform(params[param].get_name(), params[param].get_min(), params[param].get_max()))
            elif params[param].get_scale() == 'linear' and params[param].get_dist() == 'normal':
                space.append(hp.normal(params[param].get_name(), params[param].get_min(), params[param].get_max()))
            elif params[param].get_scale() == 'log' and params[param].get_dist() == 'uniform':
                space.append(hp.loguniform(params[param].get_name(), params[param].get_min(), params[param].get_max()))
            elif params[param].get_scale() == 'log' and params[param].get_dist() == 'normal':
                space.append(hp.lognormal(params[param].get_name(), params[param].get_min(), params[param].get_max()))

    return space


# def solver(model, *args, **kwargs):
#     """This high-order function may be not useful.
#
#     :param model: training model
#     :param args: hyperparameters for the model
#     :param kwargs: parameters for the model
#     :return: test error
#     """
#     _, _, test_error, _, _ = model.run_solver(*args, **kwargs)
#
#     return test_error


def combine_tracks(trials):
    """

    :param trials: trials object obtained from hyperopt
    :return: the track vector of test errors
    """
    length = len(trials.trials)
    track_valid = np.array([1.])
    track_test = np.array([1.])
    for i in range(length):
        msg1 = trials.trial_attachments(trials.trials[i])['track_valid']
        msg2 = trials.trial_attachments(trials.trials[i])['track_test']
        current_track_valid = cPickle.loads(msg1)
        current_track_test = cPickle.loads(msg2)
        current_best_valid = track_valid[-1]
        current_test = track_test[-1]
        for j in range(1, len(current_track_valid)):
            if current_track_valid[j] < current_best_valid:
                current_best_valid = current_track_valid[j]
                current_test = current_track_test[j]
                track_valid = np.append(track_valid, current_best_valid)
                track_test = np.append(track_test, current_test)
            else:
                track_valid = np.append(track_valid, current_best_valid)
                track_test = np.append(track_test, current_test)

    return track_valid, track_test
