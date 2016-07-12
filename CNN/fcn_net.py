import ipt
import mxnet as mx
from my_layer import conv_relu, maxpool
import my_utils as mu
import my_net as base

def fcn_net():
    one_one = mx.symbol.Convolution(name = 'conv11', data = base.conv10, kernel = (1,1), 
        num_filter = 1, stride = (1,1), pad = (0,0) )
    return mx.symbol.LogisticRegressionOutput(name='softmax', data=one_one)


if __name__ == '__main__':
    fcn_net()