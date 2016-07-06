import ipt, logging
import mxnet as mx
import my_utils as mu
from solver import Solver
import os
from settings import PARAMS, SOLVE

from ... import ... as net
from ... import ... as get

def train(param=PARAMS, sv=SOLVE, small=False):

    sv['name'] = 'TEST'
    input_var = raw_input('Are you testing now? ')
    
    if 'no' in input_var:
        sv.pop('name')
    else:
        sv['name'] += input_var

    out = get(6, small=True, aug=True) 
    net = net(
        use_logis=True
        )

    param['eval_data'] = out[1] 
  
    s = Solver(net, out[0], sv, **param)
    s.train()
    s.predict()
    s.all_to_png()
    s.save_best_model()
    s.plot_process()

if __name__ == '__main__':
    # temperal setting
    #SOLVE['load'] = True
    #SOLVE['load_perfix'] = '/home/zijia/HeartDeepLearning/Net/CNN/Result/<1-15:28:48>[E40]/[ACC-0.92596 E38]'
    #SOLVE['load_epoch'] = 38
    #SOLVE['use_logis'] = True
    #SOLVE['block_bn'] = True
    
    #PARAMS['num_epoch'] = 30
    # PARAMS['optimizer'] = 'adam'
    # PARAMS['learning_rate'] = 1e-2

    train()