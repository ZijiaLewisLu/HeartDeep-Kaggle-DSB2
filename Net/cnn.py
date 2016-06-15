import ipt
import mxnet as mx
import basic_right_shape as b
import Softmax as sfmx
from utils import *
import load_data as load

import os
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'

def train():
	train, val = load.get_small(1)

	net = mx.sym.Custom(data = b.out, name = 'softmax', op_type = 'sfmx')

	model = mx.model.FeedForward(
		net,
		ctx = mx.context.gpu(1),
		learning_rate = 1e-2,
		num_epoch = 5,
		)

	c = Callback()

	model.fit(
		train,
		eval_data = val,
		eval_metric = mx.metric.create(eval_iou),
		epoch_end_callback = c
		)

	c.each_to_png()

train()