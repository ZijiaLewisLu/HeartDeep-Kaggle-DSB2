import ipt
import mxnet as mx
from iou_layer import *
import experi_net as e
import numpy as np
import basic_right_shape as b
def run(sym):
    net = mx.sym.Custom(data = sym, name = 'softmax', op_type = 'iou')
    img = np.ones((5,1,256,256))
    shape = net.infer_shape(data = (5,1,256,256))[1][0]
    print 'll shape', shape
    ll = np.ones(shape)
    
    itr = mx.io.NDArrayIter(img, label = ll, batch_size = 1)
    
    model = mx.model.FeedForward(
        net,
        num_epoch = 5
    )

    shapes = itr.provide_data + itr.provide_label
    model._init_params(dict(shapes))
    print 'done init params'

    model.predict(itr, num_batch=5)

run(b.up1)
