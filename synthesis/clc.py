import ipt
import numpy as np
import mxnet as mx

from rnn.lstm import lstm_unroll
import my_utils as mu
from rnn.rnn_iter import get
from tools import get_data

from rnn.rnn_solver import Solver
from settings import PARAMS, SOLVE

def train(param=PARAMS, sv=SOLVE, small=False):
    num_hidden = 4
    num_lstm_layer = 1
    batch_size = 1


    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len, num_hidden=num_hidden, num_label=1)

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden, 256, 256)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden, 256, 256)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    
    data_train, data_val = get_data('r', batch_size, 
                    init_states=init_states, small=small)
    
    #data = get(init_states, bs=batch_size, small=small)
    #data_train = data['train']
    #data_val   = data['val']
    param['eval_data'] = data_val

    num_time = data_train.data_list[0].shape[1]
    symbol = sym_gen(num_time)
    
    s = Solver(symbol, data_train, sv, **param)
    print 'Start Training...'
    s.train()
    #s.predict()
    
   

if __name__ == '__main__':
    PARAMS['num_epoch'] = 20
    PARAMS['learning_rate'] = 0.1
    PARAMS['ctx'] = mu.gpu(1)
    
    # SOLVE['load'] = False
    # SOLVE['load_perfix'] = '/home/zijia/HeartDeepLearning/RNN/Result/<9-10:28:52>LSTM[E50]/[ACC-0.34549 E49]'
    # SOLVE['load_epoch']  = 49
    SOLVE['is_rnn'] = True

    SOLVE['name'] = 'CLC'

    train(small=False)
