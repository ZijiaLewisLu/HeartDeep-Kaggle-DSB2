
# coding: utf-8

import ipt
import mxnet as mx
from rnn.rnn_solver import Solver
import my_utils as mu
import os
import pickle as pk
import matplotlib.pyplot as plt

PARAMS={
    'ctx':mu.gpu(2),
    'learning_rate':1,
    'num_epoch':10,
    'initializer':mx.initializer.Xavier(rnd_type='gaussian'),
}

SOLVE = {
    'save_best':True,
    'is_rnn'   :False,  
}

            
from my_net import net
from tools import get_data

def cf_train( sv=SOLVE, param=PARAMS ):

    train, val = get_data('c', 2, small=False)

    sv['name'] = 'CF'
    sv['is_rnn'] = False
    param['eval_data'] = val 
    param['num_epoch'] = 20
    param['learning_rate'] = 0.1
    
    print 'SOLVE',sv
    print 'param',param
    s = Solver(net, train, sv, **param)
    s.train()
    # s.predict()
    
    return s

s = cf_train()

