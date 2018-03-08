from hyperopt import hp


def convert_params(params):
    space = []
    for param in params:
        if param.get_type() == 'integer':
            space.append(hp.quniform(param.name, param.min_val, param.max_val, param.interval))

    return space
