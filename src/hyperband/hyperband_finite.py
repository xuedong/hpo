import numpy as np
import timeit
import utils

def hyperband_finite(model, resource_type, params, min_units, max_units, runtime, path, eta=4., budget=0, n_brackets=2, s_run=None, doubling=False):
    """Hyperband with finite horizon.

    :param model: object with subroutines to generate arms and train models
    :param resource_type: type of resource to be allocated
    :param params: hyperparameter search space
    :param min_units: minimum units of resources can be allocated to one configuration
    :param max_units: maximum units of resources can be allocated to one configuration
    :param runtime: runtime patience (in min)
    :param path: path to the directory where output are stored
    :param eta: elimination proportion
    :param budget: total budget for one bracket
    :param n_brackets: number of brackets
    :param s_run: option to repeat a specific bracket
    :param doubling: option to decide whether we want to double the per bracket budget in the outer loop
    :return: None
    """
    start_time = timeit.default_timer()
    # result storage
    results = {}
    durations = []

    s = 0
    while s_to_m(timeit.default_timer()) < runtime and s < n_brackets:

