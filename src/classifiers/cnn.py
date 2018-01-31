from __future__ import print_function

import numpy as np

import theano
import theano.tensor as ts
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

from classifiers.logistic import LogisticRegression
from classifiers.mlp import HiddenLayer


class ConvPoolLayer(object):
    def __init__(self, rng, input_data, filter_shape, image_shape, pool_size):
        """This layer includes one convolution layer followed by a max-pooling layer.

        :param rng: random state
        :param input_data: input image tensor
        :param filter_shape: (nb of filters, nb of input feature maps, filter height, filter width)
        :param image_shape: (batch size, nb of input feature maps, image height, image width)
        :param pool_size: pooling/downsampling factors
        """
        assert image_shape[1] == filter_shape[1]
        self.input_data = input_data
