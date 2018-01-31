from __future__ import print_function

import numpy as np

import theano
import theano.tensor as ts
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d


class ConvPoolLayer(object):
    def __init__(self, rng, input_data, filter_shape, image_shape, pool_size):
        """This layer includes one convolution layer followed by a max-pooling layer.

        :param rng: random state
        :param input_data: input image tensor
        :param filter_shape: (nb of filters, nb of input feature maps, filter height, filter width)
        :param image_shape: (batch size, nb of input feature maps, image height, image width)
        :param pool_size: pooling/subsampling factors
        """
        assert image_shape[1] == filter_shape[1]
        self.input_data = input_data
        # initialize (nb of input feature maps) * (nb of filters) weight matrices
        # where the bounds depend on the input and output dimensions
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:])) // np.prod(pool_size)
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        w_values = np.asarray(
            rng.uniform(
                low=-w_bound,
                high=w_bound,
                size=filter_shape
            ),
            dtype=theano.config.floatX
        )
        self.w = theano.shared(value=w_values, borrow=True)
        # initialize (nb of filers) biases
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        # output after convolution and pooling
        # convolution between input feature maps and filters
        conv = conv2d(
            input=input_data,
            filters=self.w,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        # max-pooling for each feature map
        pooling = pool.pool_2d(
            input=conv,
            ds=pool_size,
            ignore_border=True
        )
        self.output = ts.tanh(pooling + self.b.dimshuffle('x', 0, 'x', 'x'))
        # parameters of the model
        self.params = [self.w, self.b]
        # keep track of the input data
        self.input_data = input_data
