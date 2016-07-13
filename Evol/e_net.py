import ipt
import mxnet as mx
import mxnet.symbol as S
from my_layer import *

data =S.Variable(name='data')   #256,256

l1_1 = conv_relu('1', data=data, kernel=(5,5), num_filter=32, pad=(2,2))
l1_2 = conv_relu('2', data=l1_1, kernel=(3,3), num_filter=32, pad=(1,1))
l1_3 = conv_relu('3', data=l1_2, kernel=(3,3), num_filter=64, pad=(1,1), bn=True)

# try overlap in future
p1 = maxpool(l1_3)              #128,128

l2_1 = conv_relu('4', data=p1,   kernel=(3,3), num_filter=64, pad=(1,1))
l2_2 = conv_relu('5', data=l2_1, kernel=(3,3), num_filter=64, pad=(1,1))
l2_3 = conv_relu('6', data=l2_2, kernel=(3,3), num_filter=64, pad=(1,1), bn=True)

p2 = maxpool(l2_3)
p1_5  = maxpool(p1)
p2 = p2+p1_5                       #64,64

l3_1 = conv_relu('7', data=p2,   kernel=(3,3), num_filter=64, pad=(1,1))
l3_2 = conv_relu('8', data=l3_1, kernel=(3,3), num_filter=16, pad=(1,1))
l3_3 = conv_relu('9', data=l3_2, kernel=(3,3), num_filter=16, pad=(1,1), bn=True)
l3_4 = S.Convolution(name='c10', data=l3_3, kernel=(1,1), num_filter=1, pad=(0,0))

pred = S.LogisticRegressionOutput(data=l3_4, name='softmax')


def e_net(*args):
    return pred


def e_rnn(*args):
    return LSTM(l3_4, 64*64, 1, 64, 64)

if __name__ == '__main__':
    for _ in pred.infer_shape(data=(11,1,256,256)):
        print _







