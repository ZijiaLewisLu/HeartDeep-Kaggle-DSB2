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
    SOLVE['load_perfix'] = '/home/zijia/HeartDeepLearning/CNN/Result/<11-21:23:29>fcn_train.py[E30]/[ACC-0.94367 E28]'
    SOLVE['load_epoch'] = 28
    #SOLVE['use_logis'] = True
    #SOLVE['block_bn'] = True
    
    PARAMS['num_epoch'] = 10
    # PARAMS['optimizer'] = 'adam'
    PARAMS['learning_rate'] = 1e-2
    #PARAMS['wd'] = 5e-6

    train()