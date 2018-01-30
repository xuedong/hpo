import theano
import theano.tensor as T

import os
import sys
sys.path.insert(0, '../src/classifiers')
sys.path.insert(0, '../src')

import logistic
import utils

#x = T.matrix('x')
#y = T.ivector('y')

#classifier = logistic.LogisticRegression(input=x, n=28*28, m=10)

#cost = classifier.neg_log_likelihood(y)
#print(cost)

#print(__file__)

data = utils.load_data('mnist.pkl.gz')
input, target = data[0]
print(input.get_value(borrow=True))
