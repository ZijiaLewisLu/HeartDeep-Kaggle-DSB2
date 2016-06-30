import ipt, logging
import mxnet as mx
from cnn import cnn_net
from my_utils import *
from solver import Solver
import os

PARAMS={
    'ctx':[mx.context.gpu(3), mx.context.gpu(2)],
    'learning_rate':3,
    'num_epoch':1,
    'optimizer':'adam',
    'initializer':mx.initializer.Xavier(rnd_type='gaussian'),
}

SOLVE = {
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
    param['num_epoch'] = 15
  
    s = Solver(net, out['train'], SOLVE, **param)
    s.train()
    s.predict()
    s.all_to_png()
    s.save_best_model()
    s.plot_process()

if __name__ == '__main__':
    main()

    
