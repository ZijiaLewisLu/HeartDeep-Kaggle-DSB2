import ipt
import mxnet as mx
import minpy as mp
import minpy.numpy as np
import minpy.numpy.random as random

def create_net(pm):
    ''' pm should be a dict of the params of each layers '''
    data = mx.sym.Variable(name= 'x')


    conv1 = mx.sym.Convolution(name = 'conv1', data = data, kernel = pm['c1']['fsize'], 
            num_filter = pm['c1']['fnum'], stride = pm['c1']['stride'], pad = pm['c1']['pad'] )
    relu1 = mx.sym.Activation(data = conv1, act_type = 'relu')
    conv2 = mx.sym.Convolution(name = 'conv2', data = relu1, kernel = pm['c2']['fsize'], 
        num_filter = pm['c2']['fnum'], stride = pm['c2']['stride'], pad = pm['c2']['pad'] )
    relu2 = mx.sym.Activation(data = conv2, act_type = 'relu')

    pool1 = mx.sym.Pooling(data = relu2, pool_type = "max", kernel=(2,2), stride = (2,2))


    conv3 = mx.sym.Convolution(name = 'conv3', data = pool1, kernel = pm['c3']['fsize'], 
            num_filter = pm['c3']['fnum'], stride = pm['c3']['stride'], pad = pm['c3']['pad'] )
    relu3 = mx.sym.Activation(data = conv3, act_type = 'relu')
    pool2 = mx.sym.Pooling(data = relu3, pool_type = "max", kernel=(2,2), stride = (2,2))
    

    conv4 = mx.sym.Convolution(name = 'conv4', data = pool2, kernel = pm['c4']['fsize'], 
            num_filter = pm['c4']['fnum'], stride = pm['c4']['stride'], pad = pm['c4']['pad'] )
    relu4 = mx.sym.Activation(data = conv4, act_type = 'relu')
    pool3 = mx.sym.Pooling(data = relu4, pool_type = "max", kernel=(2,2), stride = (2,2))


    conv5 = mx.sym.Convolution(name = 'conv5', data = pool3, kernel = pm['c5']['fsize'], 
            num_filter = pm['c5']['fnum'], stride = pm['c5']['stride'], pad = pm['c5']['pad'] )
    relu5 = mx.sym.Activation(data = conv5, act_type = 'relu')
    conv6 = mx.sym.Convolution(name = 'conv6', data = relu5, kernel = pm['c6']['fsize'], 
        num_filter = pm['c6']['fnum'], stride = pm['c6']['stride'], pad = pm['c6']['pad'] )
    relu6 = mx.sym.Activation(data = conv6, act_type = 'relu')


    up1  = mx.sym.Upsampling(data = relu6, scale = 2, sample_type = 'bilinear', num_args = 1)

    
    conv7 = mx.sym.Convolution(name = 'conv7', data = up1, kernel = pm['c7']['fsize'], 
        num_filter = pm['c7']['fnum'], stride = pm['c7']['stride'], pad = pm['c7']['pad'] )
    relu7 = mx.sym.Activation(data = conv7, act_type = 'relu')

    up2  = mx.sym.Upsampling(data = relu7, scale = 2, sample_type = 'bilinear', num_args = 1)
    
    conv8 = mx.sym.Convolution(name = 'conv8', data = up2, kernel = pm['c8']['fsize'], 
        num_filter = pm['c8']['fnum'], stride = pm['c8']['stride'], pad = pm['c8']['pad'] )
    relu8 = mx.sym.Activation(data = conv8, act_type = 'relu')

    up3  = mx.sym.Upsampling(data = relu3, scale = 2, sample_type = 'bilinear', num_args = 1)


    conv9 = mx.sym.Convolution(name = 'conv9', data = up3, kernel = pm['c9']['fsize'], 
            num_filter = pm['c9']['fnum'], stride = pm['c9']['stride'], pad = pm['c9']['pad'] )
    relu9 = mx.sym.Activation(data = conv9, act_type = 'relu')
    conv10 = mx.sym.Convolution(name = 'conv10', data = relu9, kernel = pm['c10']['fsize'], 
        num_filter = pm['c10']['fnum'], stride = pm['c10']['stride'], pad = pm['c10']['pad'] )
    relu10 = mx.sym.Activation(data = conv10, act_type = 'relu')


    conv11 = mx.sym.Convolution(name = 'conv11', data = relu10, kernel = pm['c11']['fsize'], 
            num_filter = pm['c11']['fnum'], stride = pm['c11']['stride'], pad = pm['c11']['pad'] )
    softmax = mx.sym.SoftmaxOutput(name = 'softmax', data = conv11)

    return softmax


def make_mxnet(net, mx_net = True):

    model = mx.model.FeedForward(symbol = net,
                num_epoch = 20,
                    learning_rate = .1)
    
    


    return mp.core.function(net)


if __name__ == '__main__':

    net = create_net(params)

    model = mx.model.FeedForward(symbol = net, num_epoch = 100, learning_rate = )
