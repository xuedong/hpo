import timeit
import theano.tensor as ts

import src.ho.hoo as hoo
import src.classifiers.logistic as logistic

if __name__ == '__main__':
    start = timeit.default_timer()

    x = ts.matrix('x')
    test_model = logistic.LogisticRegression(x, 28*28, 10)
    params = test_model.get_search_space()
