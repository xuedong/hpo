import unittest

import theano.tensor as ts
from hyperopt import hp

import src.classifiers.logistic as logistic
from src.bo.tpe_hyperopt import convert_params


x = ts.matrix('x')
test_model = logistic.LogisticRegression(x, 28*28, 10)
params = test_model.get_search_space()
space = [
    hp.loguniform('learning_rate', 1 * 10 ** (-3), 1 * 10 ** (-1)),
    hp.quniform('batch_size', 1, 1000, 1),
]


class TestHyperoptMethods(unittest.TestCase):

    def test_convert(self):
        self.assertEqual(convert_params(params), space)


if __name__ == '__main__':
    unittest.main()
