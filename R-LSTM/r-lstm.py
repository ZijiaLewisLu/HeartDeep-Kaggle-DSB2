import ipt
import mxnet as mx
from my_layer import conv_relu, deconv_relu

import mxnet.symbol as S
from collections import namedtuple

LSTMParam = namedtuple("LSTMParam", ["x2g_weight", "x2g_bias","h2g_weight", "h2g_bias", 'Y_weight', 'Y_bias'])

def r_lstm_step(X, num_hidden, C, c=None, h=None, idx='', param=None):

    if not isinstance(idx, str):
        idx=str(idx)
    if not c:
        c = mx.sym.Variable(name='c%s'%idx)
    if not h:
        h = mx.sym.Variable(name='h%s'%idx)
    if not param:
        param = LSTMParam(  x2g_weight= S.Variable("x2g_weight"),
                            x2g_bias=   S.Variable("x2g_bias"),
                            h2g_weight= S.Variable("h2g_weight"),
                            h2g_bias=   S.Variable("h2g_bias"),
                            Y_weight=   S.Variable("Y_weight"),
                            Y_bias=     S.Variable("Y_bias")
                            )

    x2g = S.Convolution(name='x2g%s'%idx, data=X, weight=param.x2g_weight, bias=param.x2g_bias,
                            kernel=(5,5), num_filter=num_hidden*4, pad=(2,2))
    h2g = S.Convolution(name='h2g%s'%idx, data=h, weight=param.h2g_weight, bias=param.h2g_bias,
                            kernel=(5,5), num_filter=num_hidden*4, pad=(2,2))

    gates = x2g + h2g
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4, name='rnn_slice%s'%idx)
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid", name='in_gate%s'%idx)
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh", name='in_transform%s'%idx)
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid", name='forget_gate%s'%idx)
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid", name='out_gate%s'%idx)
    
    c_this = (forget_gate * c) + (in_gate * in_transform)
    h_this = out_gate * mx.sym.Activation(c_this, act_type="tanh", name='tanh2h%s'%idx)
    
    fc = S.Convolution(name='Y%s'%idx, data=h_this, weight=param.Y_weight, bias=param.Y_bias,
                            kernel=(1,1), num_filter=C, pad=(0,0))
    c_this = mx.sym.BlockGrad(data=c_this)
    h_this = mx.sym.BlockGrad(data=h_this)

    return fc, c_this, h_this

def r_lstm(seq_len, num_hidden, C):
    T = seq_len
    cs = [S.Variable('c')]
    hs = [S.Variable('h')]
    preds  = []
    datas  = [S.Variable('data%d'%i) for i in range(T)]
    param = LSTMParam(  x2g_weight=S.Variable("x2g_weight"),
                        x2g_bias  =S.Variable("x2g_bias"),
                        h2g_weight=S.Variable("h2g_weight"),
                        h2g_bias  =S.Variable("h2g_bias"),
                        Y_weight  =S.Variable("Y_weight"),
                        Y_bias    =S.Variable("Y_bias"))
    for t in range(T):
        pred, c, h = r_lstm_step(datas[t], num_hidden, C, c=cs[-1], h=hs[-1], param=param)
        pred = S.LogisticRegressionOutput(data=pred, name='logis%d'%t)
        preds.append(pred)
        cs.append(c)
        hs.append(h)
    return S.Group(preds)


if __name__ == '__main__':
    r_lstm(30,4,1)