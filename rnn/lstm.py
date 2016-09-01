# pylint:skip-file
import sys
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math

from my_layer import conv_relu,  maxpool
from my_net import pm as Param
import mxnet.symbol as S

def gen():
    for i in range(1000):
        yield i

gtr = gen()

def cnn_forward(data, P, D):

    def C(idx, indata, bn):
        p= Param['c%d'%idx]
        i= 4*(idx-1)
    
        c= S.Convolution(name='c'+str(idx),
                data=indata,
                kernel=p['fsize'],
                num_filter=p['fnum'],
                pad=p['pad'],
                stride=p['stride'],
                weight=P[i], 
                bias=P[i+1]
                )
        if bn:
            c = mx.sym.BatchNorm(
                    name='b'+str(next(gtr)), data=c, 
                    gamma=P[i+2], beta=P[i+3],
                    )
        r = mx.sym.Activation(name='r'+str(idx), data=c, act_type='relu')
        return r
    
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
        weight=D[0]
        )

    conv7 = C(7, up1, True)
    up2   = S.Deconvolution(
        data=conv7, kernel=(4,4), stride=(2,2), pad=(1,1),
        num_filter=64, no_bias=True,
        weight=D[1] 
        )

    conv8 = C(8, up2, True)
    up3   = S.Deconvolution(
        data=conv8, kernel=(4,4), stride=(2,2), pad=(1,1),
        num_filter=32, no_bias=True,
        weight=D[2] 
        )

    conv9 = C(9, up3, True)
    conv10 = C(10, conv9, True)

    return conv10


LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.Convolution(data=indata, 
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_filter=num_hidden * 4,
                                kernel=(5,5), pad=(2,2),
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.Convolution(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_filter=num_hidden * 4,
                                kernel=(5,5), pad=(2,2),
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


# we define a new unrolling function here because the original
# one in lstm.py concats all the labels at the last layer together,
# making the mini-batch size of the label different from the data.
# I think the existing data-parallelization code need some modification
# to allow this situation to work properly
def lstm_unroll(num_lstm_layer, seq_len,
                num_hidden, num_label, dropout=0.):

    # embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    pred_all   = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    timeseq =  mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)
    labelseq = mx.sym.SliceChannel(data=label, num_outputs=seq_len, squeeze_axis=1)
    
    # CNN param
    layer_num = 10
    P = []
    for i in range(layer_num):
        P.append(S.Variable('c%d_weight'%i))
        P.append(S.Variable('c%d_bias'%i))
        P.append(S.Variable('bn%d_gamma'%i))
        P.append(S.Variable('bn%d_beta'%i))
    up_num = 3
    D = []
    for i in range(up_num):
        D.append( S.Variable('deconv%d_weight'%i))
        # D.append( S.Variable('deconv%d_bias'%i)  )

    for seqidx in range(seq_len):
        hidden = timeseq[seqidx]
        # embed in CNN
        hidden = cnn_forward(hidden, P, D)

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        #hidden_all.append(hidden)
        
        pred = mx.sym.Convolution(data=hidden, weight=cls_weight, bias=cls_bias, name='pred%d'%seqidx, 
                                    kernel=(1,1), num_filter=num_label, pad=(0,0))
        pred = mx.sym.LogisticRegressionOutput(data=pred, label=labelseq[seqidx], name='logis%d'%seqidx)
        pred_all.append(pred)
    
    # hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    # pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=num_label,
    #                              weight=cls_weight, bias=cls_bias, name='pred')

        
    ################################################################################
    # Make label the same shape as our produced data path
    # I did not observe big speed difference between the following two ways

    # label = mx.sym.transpose(data=label)
    # label = mx.sym.Reshape(data=label, target_shape=(0,))

    #label_slice = mx.sym.SliceChannel(data=label, num_outputs=seq_len)
    #label = [label_slice[t] for t in range(seq_len)]
    #label = mx.sym.Concat(*label, dim=0)
    #label = mx.sym.Reshape(data=label, target_shape=(0,))
    ################################################################################

    # sm = mx.sym.LogisticRegressionOutput(data=pred, label=label, name='softmax')

    return mx.sym.Group(pred_all)

def lstm_inference_symbol(num_lstm_layer, input_size,
                          num_hidden, num_embed, num_label, dropout=0.):
    seqidx = 0
    # embed_weight=mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                      i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                      h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                      h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)
    data = mx.sym.Variable("data")

    hidden = mx.sym.Embedding(data=data,
                              input_dim=input_size,
                              output_dim=num_embed,
                              weight=embed_weight,
                              name="embed")
    # stack LSTM
    for i in range(num_lstm_layer):
        if i==0:
            dp=0.
        else:
            dp = dropout
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[i],
                          param=param_cells[i],
                          seqidx=seqidx, layeridx=i, dropout=dp)
        hidden = next_state.h
        last_states[i] = next_state
    # decoder
    if dropout > 0.:
        hidden = mx.sym.Dropout(data=hidden, p=dropout)
    fc = mx.sym.FullyConnected(data=hidden, num_hidden=num_label,
                               weight=cls_weight, bias=cls_bias, name='pred')
    sm = mx.sym.SoftmaxOutput(data=fc, name='softmax')
    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)
