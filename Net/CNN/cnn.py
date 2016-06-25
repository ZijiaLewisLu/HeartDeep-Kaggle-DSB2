import ipt, logging
import mxnet as mx
import net as n
from utils import *

import os
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'


def batch_end(params):
    """epoch, nbatch, eval_metric, locals """

    for pairs in zip(params[3]['executor_manager'].param_names, params[3]['executor_manager'].param_arrays):
        n, p = pairs
        if 'beta' in n:
            shape = p[0].shape
            conttx = p[0].context
            p[0] = mx.ndarray.zeros(shape, ctx=conttx)


def train():
    out = get(
        1,
        # small=True
    )

    train = out['train']
    val = out['val']

    net = mx.sym.Reshape(data = n.bn10, target_shape = (0, 1*256*256))
    net = mx.sym.FullyConnected(data = net, name = 'full1', num_hidden = 100)
    net = mx.sym.FullyConnected(data = net, name = 'full2', num_hidden = 1*256*256)
    net = mx.sym.Reshape(data = net, target_shape = (0,1,256,256))
    net = mx.sym.Activation(data=net, act_type='sigmoid')
    net = mx.sym.Custom(data=net, name='softmax', op_type='sfmx')
    # net = mx.sym.LogisticRegressionOutput(data = n.out, name = 'softmax')

    model = mx.model.FeedForward(
        net,
        ctx=mx.context.gpu(1),
        learning_rate=4,
        num_epoch=50,
        optimizer='adam',
        initializer=mx.initializer.Xavier(rnd_type='gaussian')
    )

    c = Callback(draw_each=True)

    if True:
        model.fit(
            train,
            eval_data=val,
            eval_metric=mx.metric.create(c.eval),
            epoch_end_callback=c.epoch,
            batch_end_callback=c.batch,
        )

    else:
        d = train.provide_data
        l = train.provide_label
        model._init_params(dict(d + l))

    if False:
	    pred = model.predict(
	        train,
	        num_batch=4,
	        return_data=True
	    )

	    N = pred[0].shape[0]

	    # for idx in range(N):
	    # 	fig = plt.figure()

	    # 	img = pred[0][idx,0]
	    # 	print '\n\nmean>>',img.mean(), 'std>>',img.std()
	    # 	for num in range(3):
	    # 		fig.add_subplot(131+num).imshow(pred[num][idx,0])
	    # 	fig.savefig('Pred/%s-%d.png'%(c.name, idx))
	    # 	fig.clear()

    c.all_to_png()

    return c

if __name__ == '__main__':
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
	c = train()
