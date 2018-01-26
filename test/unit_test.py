import theano
import theano.tensor as T

import sys
sys.path.insert(0, '../src/classifiers')

import logistic

x = T.matrix('x')
y = T.ivector('y')

classifier = logistic.LogisticRegression(input=x, n=28*28, m=10)

cost = classifier.neg_log_likelihood(y)

print(cost)