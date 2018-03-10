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


def solver(model, *args, **kwargs):
    """This high-order function may be not useful.

    :param model: training model
    :param args: hyperparameters for the model
    :param kwargs: parameters for the model
    :return: test error
    """
    _, _, test_error, _ = model.run_solver(*args, **kwargs)

    return test_error


def combine_tracks(trials):
    """

    :param trials: trials object obtained from hyperopt
    :return: the track vector of test errors
    """
    length = len(trials.trials)
    msg = trials.trial_attachments(trials.trials[0])['track']
    track = cPickle.loads(msg)
    for i in range(1, length):
        msg = trials.trial_attachments(trials.trials[i])['track']
        current_track = cPickle.loads(msg)
        current_best = np.amin(track)
        for j in range(1, len(current_track)):
            if current_track[j] < current_best:
                current_best = current_track[j]
                track = np.append(track, current_best)
            else:
                track = np.append(track, current_best)

    return track
