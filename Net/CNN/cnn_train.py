import ipt, logging
import mxnet as mx
from cnn import cnn_net
from my_utils import *
from solver import Solver
import os
import GPU_availability as g

gpus = g.GPU_availability()[:2]

PARAMS={
    'ctx':[mx.context.gpu(i) for i in gpus],
    'learning_rate':3,
    'num_epoch':1,
    'optimizer':'adam',
    'initializer':mx.initializer.Xavier(rnd_type='gaussian'),
}

SOLVE = {
    'save_best':True,
    'is_rnn'   :False,  
}

def main(param = PARAMS, sv=SOLVE):

    sv['name'] = 'TEST'
    input_var = raw_input('Are you testing now? ')
    if 'no' in input_var:
        sv.pop('name')

    out = get(
        6,
        #small = True
    )
    
    net = cnn_net()

    param['eval_data'] = out['val'] 
  
    s = Solver(net, out['train'], sv, **param)
    s.train()
    s.predict()
    s.all_to_png()
    s.save_best_model()
    s.plot_process()

if __name__ == '__main__':
    # temperal setting
    SOLVE['load'] = True
    SOLVE['load_perfix'] = '/home/zijia/HeartDeepLearning/Net/CNN/Result/<0Save>/<1-12:30:48>[E30]/[ACC-0.90725 E29]'
    SOLVE['load_epoch '] = 29
    PARAMS['num_epoch'] = 40

    main()

    
