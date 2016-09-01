import ipt
from my_layer import conv_relu, deconv_relu, maxpool
from my_net import pm as Param
import mxnet.symbol as S

def C(idx, indata, bn):
    n=idx
    p = Param['c%d'%idx]
    
    conv = S.Convolution(name='c'+n,
            data=indata,
            kernel=p['fsize'],
            num_filter=p['fnum'],
            pad=p['pad'],
            stride=p['stride'],
            weight=P[i*2], 
            bias=P[i*2+1]
            )
    if bn:
        c = mx.sym.BatchNorm(name='b'+n, data=c)
    r = mx.sym.Activation(name='r'+n, data=c, act_type='relu')
    return r

def cnn_forward(data):

    def C(idx, indata, bn):

        return conv_relu('conv%d'%(idx), bn=bn,
            data=indata,
            kernel=p['fsize'],
            num_filter=p['fnum'],
            pad=p['pad'],
            stride=p['stride'],
            weight=P[i*2], 
            bias=P[i*2+1]
            )

    conv1 = C(1, data, True)
    conv2 = C(2, conv1, True)
    pool1 = maxpool(conv2)

    conv3 = C(3, pool1, True)
    pool2 = maxpool(conv3)

    conv4 = C(4, pool2, True)
    pool3 = maxpool(conv4)

    conv5 = C(5, pool3, True)
    conv6 = C(6, conv5, True)
    up1   = S.Deconvolution(
        data=conv6, kernel=(4,4), stride=(2,2), pad=(1,1),
        num_filter=64, no_bias=True,
        weight=dw0, bias=db0
        )

    conv7 = C(7, up1, True)
    up2   = S.Deconvolution(
        data=conv7, kernel=(4,4), stride=(2,2), pad=(1,1),
        num_filter=64, no_bias=True,
        weight=dw1, bias=db1
        )

    conv8 = C(8, up2, True)
    up3   = S.Deconvolution(
        data=relu8, kernel=(4,4), stride=(2,2), pad=(1,1),
        num_filter=32, no_bias=True,
        weight=dw2, bias=db2
        )

    conv9 = C(9, up3, True)
    conv10 = C(10, conv9, True)

    return conv10



