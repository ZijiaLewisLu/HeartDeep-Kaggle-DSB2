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
	out = get(1)

	train = out['train']
	val   = out['val']

	net = mx.sym.Custom(data = b.out, name = 'softmax', op_type = 'sfmx')

	model = mx.model.FeedForward(
		net,
		ctx = mx.context.gpu(1),
		learning_rate = 1,
		num_epoch = 100,
		)

	c = Callback()

	# for i in range(5):
	model.fit(
		train,
		eval_data = val,
		eval_metric = mx.metric.create(eval_iou),
		epoch_end_callback = c
		)

        pred =  model.predict(
		val,
		num_batch = 5,
		return_data = True
		)
        
        N = pred[0].shape[0]
        for idx in range(N):
            fig = plt.figure()
            for num in range(3):
                fig.add_subplot(131+num).imshow(pred[num][idx,0])
            fig.savefig('Pred/%s-%d.png'%(c.name, idx))



#	c.each_to_png()

	return c

if __name__ == '__main__':
	c = train()
	
