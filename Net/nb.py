import ipt
import mxnet as mx
from iou_layer import *
# import experi_net as e
import numpy as np
import basic_right_shape as b
import load_data as load
import matplotlib.pyplot as plt


import logging
# logging.basicConfig(
#     format='%(levelname)s:%(message)s', 
#     level=logging.DEBUG)

class Callback():

    def __init__(self, name = None):
        self.loss_hist = {}
        self.name = name

    def __call__(self, epoch, symbol, arg_params, aux_params, loss):
        self.loss_hist[epoch] = loss
        print 'Epoch[%d] Train accuracy: %f' % ( epoch, np.sum(loss)/float(len(loss)) )
        # print symbol 
        # print arg_params.keys() 
        # print aux_params, '\n\n\n\n\n'

    def get_dict(self):
        return  self.loss_hist
    
    def get_list(self):
        l = []
        for k in sorted(self.loss_hist.keys()):
            print k
            l += self.loss_hist[k]
        return l

    def each_to_png(self):
        prefix = '' if self.name  == None else self.name

        for k in sorted(self.loss_hist.keys()):
            plt.plot(self.loss_hist[k])
            plt.savefig(prefix+str(k)+'.png')


    def reset(self):
        self.loss_hist = {}


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
        learning_rate = 3e-1,
        num_epoch = 10,
        ctx = mx.context.gpu(1),
        optimizer = 'adam',
        initializer = mx.initializer.Xavier(rnd_type = 'gaussian')
    )

    # shapes = train.provide_data + train.provide_label
    # model._init_params(dict(shapes))
    # print 'done init params'

    # model.predict(train, num_batch=5)

    c = Callback()

    model.fit(
    	train,
    	eval_data = val,
        batch_end_callback = mx.callback.ProgressBar(51),
        epoch_end_callback = c,
        eval_metric = mx.metric.create(eval_iou)
    	)

    c.each_to_png()



run(b.out)
