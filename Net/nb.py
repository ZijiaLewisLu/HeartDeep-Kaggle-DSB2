import ipt
import mxnet as mx
from iou_layer import *
# import experi_net as e
import numpy as np
import basic_right_shape as b
import load_data as load

def call(l):
	print l[:2]

def run(sym):
    # img = np.ones((5,1,256,256))
    # shape = net.infer_shape(data = (5,1,256,256))[1][0]
    # print 'll shape', shape
    # print net.list_arguments()
    # ll = np.ones(shape)
    
    # itr = mx.io.NDArrayIter(img, label = ll, batch_size = 1)
    train, val = load.get_()
    net = mx.sym.Custom(data = sym, name = 'softmax', op_type = 'iou')
    model = mx.model.FeedForward(
        net,
        num_epoch = 10,
        ctx = mx.context.gpu(0)
    )

    # shapes = train.provide_data + train.provide_label
    # model._init_params(dict(shapes))
    # print 'done init params'

    # model.predict(train, num_batch=5)

    model.fit(
    	train,
    	eval_data = val,
    	)




run(b.out)
