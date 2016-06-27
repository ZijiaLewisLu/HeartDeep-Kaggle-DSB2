import ipt, logging
import mxnet as mx
from cnn import cnn_net
from utils import *
from solver import Solver
import os

PARAMS={
    'ctx':[mx.context.gpu(1), mx.context.gpu(0)],
    'learning_rate':6,
    'num_epoch':200,
    'optimizer':'adam',
    'initializer':mx.initializer.Xavier(rnd_type='gaussian'),

    'save_best':True,
    'is_rnn'   :False,
}




def main():

    out = get(
        6,
    )
    net = cnn_net()

    PARAMS['eval_data'] = out['val']
    PARAMS['name'] = 'test'

 
    s = Solver(net, out['train'], **PARAMS)
    s.train()
    s.save_best()


if __name__ == '__main__':
    main()

    
