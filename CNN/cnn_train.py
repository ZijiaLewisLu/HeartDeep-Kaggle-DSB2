import ipt, logging
import mxnet as mx
from cnn import cnn_net
import my_utils as u
from solver import Solver
import os

PARAMS={
    'ctx':u.gpu(2),
    'learning_rate':3,
    'num_epoch':15,
    #'optimizer':'adam',
    'initializer':mx.initializer.Xavier(rnd_type='gaussian'),
}

SOLVE = {
    'save_best':True,
    'is_rnn'   :False,  
}

def train(param = PARAMS, sv=SOLVE, small=False):

    sv['name'] = 'TEST'
    input_var = raw_input('Are you testing now? ')
    
    if 'no' in input_var:
        sv.pop('name')
    else:
        sv['name'] += input_var


    out = u.get(6,small=True, aug=True) 
    net = cnn_net(
        use_logis=False
        )

    param['eval_data'] = out['val'] 
  
    s = Solver(net, out['train'], sv, **param)
    s.train()
    s.predict()

if __name__ == '__main__':
    # temperal setting
    #SOLVE['load'] = True
    #SOLVE['load_perfix'] = '/home/zijia/HeartDeepLearning/Net/CNN/Result/<1-15:28:48>[E40]/[ACC-0.92596 E38]'
    #SOLVE['load_epoch'] = 38
    #SOLVE['use_logis'] = True
    #SOLVE['block_bn'] = True
    
    PARAMS['num_epoch'] = 2
    # PARAMS['optimizer'] = 'adam'
    # PARAMS['learning_rate'] = 1e-2

    train(small=True)

    
