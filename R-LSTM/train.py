import ipt
from solver import Solver
import mxnet as mx
from RNN.rnn_load import get
net = __import__('r-lstm')
print net 
from CNN.cnn import cnn_net
from settings import PARAMS, SOLVE

def make_net():
    pred, c, h = net.r_lstm_step(cnn_net(), num_hidden=3, C=1)
    pred = mx.sym.LogisticRegressionOutput(data=pred, name='logis')
    sym = mx.symbol.Group([pred, c, h])
    return sym

def train(param=PARAMS, sv=SOLVE, small=False):

    net = make_net()

    out = get(2, rate=0.2, small=True) 
    train, param['eval_data'] = out['train'], out['val']  
    param['marks'] = param['e_marks'] = out['marks'] 

    s = Solver(net, train, sv, **param)
    s.train()
    s.predict()

if __name__ == '__main__':
    SOLVE['is_rnn'] = True
    train()
