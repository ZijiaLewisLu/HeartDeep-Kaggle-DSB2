# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import ipt
import numpy as np
import mxnet as mx

import my_utils as mu
from rnn.rnn_solver import Solver
from c_lstm import lstm_unroll

from tools import get_data
from settings import PARAMS, SOLVE

def train(batch_size, param=PARAMS, sv=SOLVE, small=False):
    num_lstm_layer = 1
    num_hidden = 1000
    
    # prepare data
    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    data_train, data_val = get_data('r', batch_size, 
                    init_states=init_states, small=small, splite_rate=0.2)
    param['eval_data'] = data_val

    # prepare symbol
    num_time = data_train.data_list[0].shape[1]
    symbol = lstm_unroll(num_lstm_layer, num_time, num_hidden)
    
    s = Solver(symbol, data_train, sv, **param)
    print 'Start Training...'
    s.train()
    # s.predict()
    

if __name__ == '__main__':
    
    PARAMS['learning_rate'] = 2
    
    # SOLVE['load'] = False
    # SOLVE['load_perfix'] = '/home/zijia/HeartDeepLearning/RNN/Result/<9-10:28:52>LSTM[E50]/[ACC-0.34549 E49]'
    # SOLVE['load_epoch']  = 49

    SOLVE['name'] = 'CL'
    SOLVE['is_rnn'] = True

    if False:
        PARAMS['num_epoch'] = 2
        PARAMS['ctx'] = mu.gpu(1)
        train(1, small=True)
    else:
        PARAMS['num_epoch'] = 20
        PARAMS['ctx'] = mu.gpu(2)
        train(2, small=False)

