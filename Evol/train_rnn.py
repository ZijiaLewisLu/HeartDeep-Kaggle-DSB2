import ipt, logging
import mxnet as mx
import my_utils as mu
from solver import Solver
import os
from settings import PARAMS, SOLVE

import e_net
from load_e import get_rnn as get

def train(param=PARAMS, sv=SOLVE, small=False):

    sv['name'] = __file__.rstrip('.py')
    input_var = raw_input('Are you testing now? ')
    
    if 'no' in input_var:
        sv.pop('name')
    else:
        sv['name'] += input_var

    out = get(1) 
    from my_layer import LSTM
    sym = LSTM(e_net.l3_4, 64*64, 1, 64, 64)
    sym = list(sym)
    sym[0] = mx.sym.LogisticRegressionOutput(data=sym[0], name='softmax')
    sym = mx.symbol.Group(list(sym))

    param['eval_data'] = out['val'] 
    param['marks'] = param['e_marks'] = out['marks'] 
    param['ctx'] = mu.gpu(1)

    print out['train'].label[0][1].shape
  
    s = Solver(sym, out['train'], sv, **param)
    s.train()
    s.predict()

if __name__ == '__main__':
    # temperal setting
    SOLVE['load'] = True
    SOLVE['load_perfix'] = "Result/<13-14:13:16>train_rnn[E5]/[ACC-0.70691 E4]"
    SOLVE['load_epoch']  = 4
    #SOLVE['use_logis'] = True
    #SOLVE['block_bn'] = True
    SOLVE['is_rnn'] = True
    
    PARAMS['num_epoch'] = 5
    # PARAMS['optimizer'] = 'adam'
    PARAMS['learning_rate'] = 0.5
    #PARAMS['wd'] = 5e-5

    train()