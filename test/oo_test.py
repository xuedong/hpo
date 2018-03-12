import os
import timeit
import numpy as np
import theano.tensor as ts

import src.classifiers.logistic as logistic
import src.utils as utils
import src.ho.hoo as hoo
import src.ho.hct as hct
import src.ho.target as target


if __name__ == '__main__':
    start = timeit.default_timer()

    x = ts.matrix('x')
    test_model = logistic.LogisticRegression(x, 28*28, 10)
    params = test_model.get_search_space()
