import ipt
import mxnet as mx
import net as n
from utils import *
from rnn_feed import Feed
from rnn_iter import RnnIter
import matplotlib.pyplot as plt
from utils import Callback

def rnn(dropout=0.):
    
    begin = n.reshape1 # N, 1*256*256
    num_hidden=250

    if dropout > 0.:
        begin = mx.sym.Dropout(data=begin, p=dropout)
    
    c=mx.sym.Variable(name='c')
    h=mx.sym.Variable(name='h')

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
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="rnn_slice")

    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")

    c = (forget_gate * c) + (in_gate * in_transform)
    h = out_gate * mx.sym.Activation(c, act_type="tanh")
    c = mx.sym.BlockGrad(data = c)
    h = mx.sym.BlockGrad(data = h)

    if dropout > 0.:
        h = mx.sym.Dropout(data=h, p=dropout)

    fc = mx.sym.FullyConnected(
    		data=h, 
    		num_hidden=1*256*256,
            name='pred'
            )

    reshape2 = mx.sym.Reshape(data = fc, target_shape = (0,1,256,256))
    
    if False:
        sgmd = mx.sym.Activation(data = reshape2, act_type = 'sigmoid')
        net = mx.sym.Custom(data = sgmd, name = 'softmax', op_type = 'sfmx')
        # net = mx.sym.MakeLoss(data = net, name='loss')
    else:
        net = mx.sym.LogisticRegressionOutput(data = reshape2, name = 'softmax')
    
    group = mx.sym.Group([net,c,h])
    return group

def contruct_iter():
    img, ll, _, _ = load_pk("/home/zijia/HeartDeepLearning/Net/data/o1.pk")
    print img.shape

    img = img[:60]
    img = img.reshape((30,2,1,256,256))
    ll  = ll[:60]
    ll  = ll.reshape((30,2,1,256,256))

    return RnnIter(img,ll)

if __name__ == '__main__':


    net = rnn()

    train = contruct_iter()

    # for b in train:
    #     print b

    # train.reset()

    # for b in train:
    #     print b

    # assert False

    marks = np.ones((30)).astype('int')

    c = Callback()
    
    model = Feed(
        net,
        #rnn_hidden
        ctx = mx.context.gpu(0),
        learning_rate = 1.5,
        num_epoch = 10
        )

    model.fit(
        train,
        marks,
        eval_data = train,
        e_marks = marks,
        eval_metric = RnnM(eval_iou),
        epoch_end_callback = c.epoch,
        batch_end_callback = c.batch,
        )

    c.all_to_png()

