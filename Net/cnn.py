import ipt
import mxnet as mx
import net as b
# import Softmax as sfmx							
from utils import *
# import load_data as load

import os
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'

def batch_end(params):
	'''epoch, nbatch, eval_metric, locals '''

	print params['locals']
	assert False



def train():
	out = get(1, small = True)

	train = out['train']
	val   = out['val']

	net = mx.sym.Custom(data = b.out, name = 'softmax', op_type = 'sfmx')

	model = mx.model.FeedForward(
		net,
		ctx = mx.context.gpu(1),
		learning_rate = 1,
		num_epoch = 1,
		)

	c = Callback()

	# for i in range(5):
	model.fit(
		train,
		eval_data = val,
		eval_metric = mx.metric.create(eval_iou),
		epoch_end_callback = c
		)

		# model.predict(
		# 	val,
		# 	num_batch = 1,
		# 	return_data = True
		# 	)

	c.each_to_png()

	return c

if __name__ == '__main__':
	c = train()
	