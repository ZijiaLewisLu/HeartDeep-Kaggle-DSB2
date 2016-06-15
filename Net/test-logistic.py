import ipt
import mxnet as mx
import basic_right_shape as b
import load_data as load
from utils import *
import matplotlib.pyplot as plt
# import create_train_modle as ctm


net = mx.sym.LogisticRegressionOutput(data = b.out, name = 'softmax')

def train_l():

	train, val = load.get_(1)

	c = Callback()

	model = mx.model.FeedForward(
		net,
		learning_rate = 1e-1,
		ctx = mx.context.gpu(1),
        num_epoch = 10
		)

	print 'start to train ...'
	model.fit(
		train,
		eval_data = val,
        eval_metric = mx.metric.create(eval_iou), 
        # num_epoch = 1000,
        epoch_end_callback = c,
        batch_end_callback = mx.callback.ProgressBar(1000)
		)

	c.each_to_png()

if __name__ == '__main__':
	train_l()