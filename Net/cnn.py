import ipt
import mxnet as mx
import net as n
# import Softmax as sfmx							
from utils import *
# import load_data as load

import os
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'

def batch_end(params):
	'''epoch, nbatch, eval_metric, locals '''
	# o = params[3]['arg_params']['batchnorm9_beta'].asnumpy().mean()  # params[3]['arg_params']['batchnorm9_gamma'].asnumpy()
	# print params[3]['arg_params']['batchnorm9_beta'].shape

	for pairs in zip(params[3]['executor_manager'].param_names, params[3]['executor_manager'].param_arrays):
		n, p = pairs
		if 'beta' in n:
			shape = p[0].shape
			conttx = p[0].context
			p[0] = mx.ndarray.zeros(shape, ctx = conttx)

	# assert False

	# if o != 0.0:
		# print 'BETA ERROR', o
		# params[3]['arg_params']['batchnorm9_beta'] = mx.ndarray.zeros((1,))
	# print '>> beta',o, params[3]['arg_params']['batchnorm9_beta'].asnumpy()
	# assert False


def train():
	out = get(
		1,
	 	small = True
	 )

	train = out['train']
	val   = out['val']

	net = mx.sym.Custom(data = n.out, name = 'softmax', op_type = 'sfmx')
	# net = mx.sym.LogisticRegressionOutput(data = n.out, name = 'softmax')


	model = mx.model.FeedForward(
		net,
		ctx = mx.context.gpu(1),
		learning_rate = 1.5,
		num_epoch = 100,
		optimizer='adam', 
		initializer = mx.initializer.Xavier(rnd_type = 'gaussian')
	)

	c = Callback()

	if True:
		model.fit(
			train,
			eval_data = val,
			# eval_metric = mx.metric.create(eval_iou),
			eval_metric = 'acc',
			epoch_end_callback = c,
			batch_end_callback = batch_end
			)

	else:
		d = train.provide_data
		l = train.provide_label
		model._init_params(dict(d+l))


		# for i in range(10):
		# 	print '\n', i,\
		# 	'beta',  model.arg_params['batchnorm%d_beta'% i].asnumpy().mean(), \
		# 	'gamma', model.arg_params['batchnorm%d_gamma' % i].asnumpy().mean()

		# print 'beta init      ',model.arg_params['batchnorm9_beta'].asnumpy()



	pred = model.predict(
		train,
		num_batch = 4,
		return_data = True
	)

	# for i in range(10):
	# 		print '\n', i
	# 		print 'beta',  model.arg_params['batchnorm%d_beta'% i].asnumpy().mean()
	# 		print 'gamma', model.arg_params['batchnorm%d_gamma' % i].asnumpy().mean()

	N = pred[0].shape[0]


	for idx in range(N):
		fig = plt.figure()

		img = pred[0][idx,0]
		print '\n\nmean>>',img.mean(), 'std>>',img.std()
		for num in range(3):
			fig.add_subplot(131+num).imshow(pred[num][idx,0])
		fig.savefig('Pred/%s-%d.png'%(c.name, idx))
		fig.clear()



	c.all_to_png()

	return c

if __name__ == '__main__':
	c = train()
	
