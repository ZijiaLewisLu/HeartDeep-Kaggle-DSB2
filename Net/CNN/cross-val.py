import ipt
import mxnet as mx
from utils import *
from cnn import cnn_net
import logging

def test():
    net   = cnn_net()

    data  = get(5, small = True)
    train = data['train']
    val   = data['val']

    for l in [ 1e-3, 3e-3, 6e-3, 1e-2, 3e-2, 6e-2, 1e-1, 3e-1, 6e-1, 1, 3, 6]:
        logging.info('------------------------------------%f-------------------------------', l)
        c = Callback(name=str(l))
        model = mx.model.FeedForward.create(
		  net,
		  train,
          learning_rate = l,
          ctx = [mx.context.gpu(i) for i in [0,1,2]],
		  eval_data = val,
		  eval_metric = mx.metric.create(c.eval),
		  num_epoch = 40,
          )

        c.all_to_png()

        predict_test(model, val, c.path)

if __name__ == '__main__':
    logging.basicConfig(filename='%s-eval.log'% parse_time(), level=logging.INFO)
    test()


