import ipt, logging
import mxnet as mx
import my_utils as mu
from solver import Solver
import os
from settings import PARAMS, SOLVE
from my_utils import get
import my_net as base

def net():
    one_one = mx.symbol.Convolution(name = 'conv11', data = base.conv10, kernel = (1,1), 
        num_filter = 1, stride = (1,1), pad = (0,0) )
    return mx.symbol.LogisticRegressionOutput(name='softmax', data=one_one)


def train(param=PARAMS, sv=SOLVE, small=False):

    sv['name'] = __file__
    input_var = raw_input('Are you testing now? ')
    
    if 'no' in input_var:
        sv.pop('name')
    else:
        sv['name'] += input_var

    out = get(6, 
        #small=True, 
        aug=True)
    global net 
    net = net()

    param['eval_data'] = out['val'] 
  
    s = Solver(net, out['train'], sv, **param)
    s.train()
    s.predict()

if __name__ == '__main__':
    # temperal setting
    SOLVE['load'] = True
    SOLVE['load_perfix'] = '/home/zijia/HeartDeepLearning/CNN/Result/<12-12:00:44>fcn_train.py[E15]/[ACC-0.93389 E14]'
    SOLVE['load_epoch'] = 14
    #SOLVE['use_logis'] = True
    #SOLVE['block_bn'] = True
    
    PARAMS['num_epoch'] = 15
    # PARAMS['optimizer'] = 'adam'
    PARAMS['learning_rate'] = 0.1
    PARAMS['wd'] = 1e-6

    train()