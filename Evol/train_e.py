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
    #SOLVE['load'] = True
    #SOLVE['load_perfix'] = '/home/zijia/HeartDeepLearning/Evol/Result/<8-01:28:54>TEST[E20]/[ACC-0.04717 E0]'
    #SOLVE['load_epoch'] = 
    #SOLVE['use_logis'] = True
    #SOLVE['block_bn'] = True
    
    PARAMS['num_epoch'] = 20
    # PARAMS['optimizer'] = 'adam'
    PARAMS['learning_rate'] = 5e-1

    train()