import ipt, logging
import mxnet as mx
from cnn import cnn_net
from my_utils import *
from solver import Solver
import os

PARAMS={
    'ctx':[mx.context.gpu(1), mx.context.gpu(0)],
    'learning_rate':3,
    'num_epoch':1,
    'optimizer':'adam',
    'initializer':mx.initializer.Xavier(rnd_type='gaussian'),

    'save_best':True,
    'is_rnn'   :False,
}

def main(param = PARAMS):

    #logging.basicConfig(level=logging.INFO)

    out = get(
        6,
        #small = True
    )
    
    net = cnn_net()

    param['eval_data'] = out['val'] 
    param['num_epoch'] = 30
  
    s = Solver(net, out['train'], **param)
    s.train()
    s.all_to_png()
    s.save_best_model()
    s.predict()
    s.plot_process()

if __name__ == '__main__':
    main()

    
