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
    """

    :param model: training model
    :param args: hyperparameters for the model
    :param kwargs: parameters for the model
    :return: test error
    """
    _, _, test_error, _ = model.run_solver(*args, **kwargs)

    return test_error
