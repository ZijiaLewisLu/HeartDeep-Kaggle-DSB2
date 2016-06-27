import ipt, logging
import mxnet as mx
import net as n
from utils import *

import os
# os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'


def batch_end(params):
    """epoch, nbatch, eval_metric, locals """

    for pairs in zip(params[3]['executor_manager'].param_names, params[3]['executor_manager'].param_arrays):
        n, p = pairs
        if 'beta' in n:
            shape = p[0].shape
            conttx = p[0].context
            p[0] = mx.ndarray.zeros(shape, ctx=conttx)


def cnn_net():
	net = mx.sym.Reshape(data = n.bn10, target_shape = (0, 1*256*256))
	net = mx.sym.FullyConnected(data = net, name = 'full1', num_hidden = 100)
	net = mx.sym.FullyConnected(data = net, name = 'full2', num_hidden = 1*256*256)
	net = mx.sym.Reshape(data = net, target_shape = (0,1,256,256))
	# net = mx.sym.Activation(data=net, act_type='sigmoid')
	# net = mx.sym.Custommnp(data=net, name='softmax', op_type='sfmx')
	net = mx.sym.LogisticRegressionOutput(data = net, name = 'softmax')
	return net

def train():
    out = get(
        6,
        # small=True
    )

    train = out['train']
    val = out['val']

    net = cnn_net()
    model = mx.model.FeedForward(
        net,
        ctx=[mx.context.gpu(1), mx.context.gpu(0),mx.context.gpu(2)],
        learning_rate=6,
        num_epoch=200,
        optimizer='adam',
        initializer=mx.initializer.Xavier(rnd_type='gaussian')
    )

    if True:
    	
    	c_train = Callback(save_best=True)

    	log_name = c_train.path + 'log.txt'
    	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO,filename=log_name)


        model.fit(
            train,
            eval_data=val,
            eval_metric=mx.metric.create(c_train.eval),
            epoch_end_callback=c_train.epoch,
            batch_end_callback=c_train.batch,
        )
    	
    	c_train.all_to_png()
    	c_train.save_best_model()

    else:
        
        d = train.provide_data
        l = train.provide_label
        model._init_params(dict(d + l))

    if True:
	    
	    out = model.predict(
	        val,
	        # num_batch=4,
	        return_data=True
	    )

	    if 'c_train' in locals().keys():
	    	prefix = c_train.path

	    else:
	    	prefix = parse_time()
	    	os.mkdir(prefix)

	    N = out[0].shape[0]

	    for idx in range(N):
	    	gap = np.ones((256,5))
	    	pred= out[0][idx,0]
	    	img = out[1][idx,0]
	    	label = out[2][idx,0]
	    	png = np.hstack([pred,gap,label])
	    	
	    	logging.debug('Pred mean>>',pred.mean(), 'std>>',pred.std())

	    	fig = plt.figure()
	    	fig.add_subplot(121).imshow(png)
	    	fig.add_subplot(122).imshow(img)
	    	fig.savefig(prefix+'Pred%d.png'%(idx))
	    	fig.clear()
	    	plt.close('all')

    return c_train

if __name__ == '__main__':
	c = train()
