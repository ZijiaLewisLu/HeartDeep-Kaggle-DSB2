import ipt
import logging
import mxnet as mx
import HeartDeepLearning.my_net as n
from HeartDeepLearning.my_utils import *
from rnn_feed import Feed
from rnn_iter import RnnIter
import matplotlib.pyplot as plt
from rnn_metric import RnnM
import mxnet.symbol as S

def rnn_net(dropout=0., logistic=True, begin=None, num_hidden=250):

    if begin is None:
        begin = n.reshape1  # N, 1*256*256

    if dropout > 0.:
        begin = mx.sym.Dropout(data=begin, p=dropout)

    c = mx.sym.Variable(name='c')
    h = mx.sym.Variable(name='h')

    i2h = mx.sym.FullyConnected(data=begin,
                                # weight=param.i2h_weight,
                                # bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="rnn_i2h")

    h2h = mx.sym.FullyConnected(data=h,
                                # weight=param.h2h_weight,
                                # bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="rnn_h2h")
    gates = i2h + h2h
    #gates._set_attr(name='gates')
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="rnn_slice")

    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid", name='in_gate')
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh", name='in_transform')
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid", name='forget_gate')
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid", name='out_gate')

    c_this = (forget_gate * c) + (in_gate * in_transform)
    #c._set_attr(name='thisC')
    h_this = out_gate * mx.sym.Activation(c_this, act_type="tanh", name='tanh2h')

    if dropout > 0.:
        h_this = mx.sym.Dropout(data=h_this, p=dropout)

    fc = mx.sym.FullyConnected(
        data=h_this,
        num_hidden=1 * 256 * 256,
        name='pred'
    )

    reshape2 = mx.sym.Reshape(data=fc, target_shape=(0, 1, 256, 256), name='reshape2')
    c_this = mx.sym.BlockGrad(data=c_this)
    h_this = mx.sym.BlockGrad(data=h_this)

    if not logistic:
        sgmd = mx.sym.Activation(data=reshape2, act_type='sigmoid')
        net = mx.sym.Custom(data=sgmd, name='softmax', op_type='sfmx')
        # net = mx.sym.MakeLoss(data = net, name='loss')
    else:
        net = mx.sym.LogisticRegressionOutput(data=reshape2, name='softmax')

    group = mx.sym.Group([net, c_this, h_this])
    return group


def contruct_iter():
    import pickle as pk
    fname = '/home/zijia/HeartDeepLearning/Net/patience/SC-HF-I-1.pk'
    with open(fname, 'r') as f:
        img = pk.load(f)
        ll = pk.load(f)
        m = pk.load(f)

    img = img[:60, None, None, :, :]
    ll = ll[:60, None, None, :, :]
    m = m[:60]

    img -= img.mean().astype('int64')

    return RnnIter(img, ll), m


def unroll_lstm(seq_len, num_hidden, C, H, W):
    from my_layer import LSTM
    T = seq_len
    cs = [S.Variable('c_init')]
    hs = [S.Variable('h_init')]
    preds  = []
    datas  = [S.Variable('data%d'%i) for i in range(T)]
    for t in range(T):
        pred, c, h = LSTM(datas[t], num_hidden, C, H, W, c=cs[-1], h=hs[-1])
        pred = S.LogisticRegressionOutput(data=pred, name='logis%d'%t)
        preds.append(pred)
        cs.append(c)
        hs.append(h)
    return S.Group(preds)


if __name__ == '__main__':
    #logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    net = rnn_net()
    train, marks = contruct_iter()
    logging.debug(marks)

    logging.info(c.name)

    num_epoch = 30
    model = Feed(
        net,
        # rnn_hidden
        ctx=mx.context.gpu(0),
        learning_rate=3,
        num_epoch=num_epoch,
    )

    model.fit(
        train,
        marks,
        # eval_data = train,
        # e_marks = marks,
        eval_metric=RnnM(c.eval),
        epoch_end_callback=c.epoch,
        batch_end_callback=c.batch,
    )


    model.save(c.path, num_epoch)

    c.all_to_png()

