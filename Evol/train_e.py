import ipt, logging
import mxnet as mx
import my_utils as mu
from solver import Solver
import os
from settings import PARAMS, SOLVE

from e_net import e_net as net
from load_e import get

def train(param=PARAMS, sv=SOLVE, small=False):

    sv['name'] = __file__.rstrip('.py')
    input_var = raw_input('Are you testing now? ')
    
    if 'no' in input_var:
        sv.pop('name')
    else:
        sv['name'] += input_var

    out = get(6, aug=True) 
    sym = net()

    param['eval_data'] = out['val'] 
  
    s = Solver(sym, out['train'], sv, **param)
    s.train()
    s.predict()

if __name__ == '__main__':
    # temperal setting
    #SOLVE['load'] = True
    #SOLVE['load_perfix'] = "/home/zijia/HeartDeepLearning/Evol/Result/<12-16:50:28>TEST[E10]/[ACC-0.80719 E9]"
    #SOLVE['load_epoch'] = 9
    #SOLVE['use_logis'] = True
    #SOLVE['block_bn'] = True
    
    PARAMS['num_epoch'] = 20
    # PARAMS['optimizer'] = 'adam'
    PARAMS['learning_rate'] = 0.5
    PARAMS['wd'] = 5e-5

    train()