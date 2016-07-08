import ipt
import mxnet as mx
import mxnet.symbol as S
from my_layer import *

def conv_relu(base_name, bn=False, **kwargs):
    n=base_name
    c = S.Convolution(name='c'+n, **kwargs)
    if bn:
        c = S.BatchNorm(name='b'+n, data=c)
    r = S.Activation(name='r'+n, data=c, act_type='relu')
    
    return r

def maxpool(data):
    return S.Pooling(data=data, pool_type="max", kernel=(2,2), stride=(2,2))



data =S.Variable(name='data')

l1_1 = conv_relu('1_1', data=data, kernel=(5,5), num_filter=32, pad=(2,2))
l1_2 = conv_relu('1_2', data=l1_1, kernel=(3,3), num_filter=32, pad=(1,1))
l1_3 = conv_relu('1_3', data=l1_2, kernel=(3,3), num_filter=64, pad=(1,1), bn=True)

p1 = maxpool(l1_3) # try overlap in future
x1 = maxpool(data)
c1 = S.Concat(p1,x1,num_args=2)

l2_1 = conv_relu('2_1', data=c1,   kernel=(3,3), num_filter=64, pad=(1,1))
l2_2 = conv_relu('2_2', data=l2_1, kernel=(3,3), num_filter=64, pad=(1,1), bn=True)
l2_3 = conv_relu('2_3', data=l2_2, kernel=(3,3), num_filter=64, pad=(1,1), bn=True)

c2 = S.Concat(l2_3,p1,x1, num_args=3)
p2 = maxpool(c2)

l3_1 = conv_relu('3_1', data=p2,   kernel=(3,3), num_filter=64, pad=(1,1), bn=True)
l3_2 = conv_relu('3_2', data=l3_1, kernel=(3,3), num_filter=16, pad=(1,1), bn=True)
l3_3 = S.Convolution(name='c3_3', data=l3_2, kernel=(3,3), num_filter=1,  pad=(1,1))

pred = S.LogisticRegressionOutput(data=l3_3, name='softmax')


def e_net(*args):
    return pred

if __name__ == '__main__':
    print pred.infer_shape(data=(11,1,256,256))







