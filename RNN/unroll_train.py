import ipt
import mxnet as mx
import my_utils as mu
from solver import Solver
from settings import PARAMS, SOLVE
import numpy as np
from  unroll import UnrollIter, unroll_lstm

def train(param=PARAMS, sv=SOLVE, small=False):
    # prepare net
    net = unroll_lstm(10, 64*64, 1, 64, 64)

    # prepare data
    from Evol.load_e import reshape_label
    from RNN.rnn_load import load_rnn_pk
    img, ll = load_rnn_pk(['../DATA/PK/NEW/[T10,N10]<8-11:42:11>.pk'])

    ll = reshape_label(ll)
    lt, lv = ll[:8], ll[8:]
    train = UnrollIter(lt, label=lt, batch_size=2, num_hidden=64*64)
    val   = UnrollIter(lv, label=lv, batch_size=2, num_hidden=64*64)

    # train
    s = Solver(net, train, sv, **param)
    print 'Start Training...'
    s.train()
    s.predict()

if __name__ == '__main__':
    PARAMS['num_epoch'] = 10
    PARAMS['learning_rate'] = 1
    PARAMS['ctx'] = mu.gpu(1)
    
    # SOLVE['load'] = False
    # SOLVE['load_perfix'] = '/home/zijia/HeartDeepLearning/RNN/Result/<9-10:28:52>LSTM[E50]/[ACC-0.34549 E49]'
    # SOLVE['load_epoch']  = 49

    SOLVE['name'] = __file__

    train(small=True)
