import ipt, logging
import mxnet as mx
import my_utils as mu
from solver import Solver
import os
from settings import PARAMS, SOLVE

from e_net import e_net as net
from load_e import get

def train(param=PARAMS, sv=SOLVE, small=False):

    sv['name'] = 'TEST'
    input_var = raw_input('Are you testing now? ')
    
    if 'no' in input_var:
        sv.pop('name')
    else:
        sv['name'] += input_var

    out = get(6, small=True, aug=False) 
    sym = net()

    param['eval_data'] = out['val'] 
  
    s = Solver(sym, out['train'], sv, **param)
    s.train()
    s.predict()
    s.all_to_png()
    s.save_best_model()
    s.plot_process()

if __name__ == '__main__':
    # temperal setting
    SOLVE['load'] = True
    SOLVE['load_perfix'] = "Result/<8-10:10:18>TEST[E50]/[ACC-0.69643 E49]"
    SOLVE['load_epoch'] = 49
    #SOLVE['use_logis'] = True
    #SOLVE['block_bn'] = True
    
    PARAMS['num_epoch'] = 50
    # PARAMS['optimizer'] = 'adam'
    PARAMS['learning_rate'] = 1

    train()