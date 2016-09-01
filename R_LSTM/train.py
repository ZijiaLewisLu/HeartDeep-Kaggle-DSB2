import ipt
from solver import Solver
import mxnet as mx
R = __import__('r_lstm')
from settings import PARAMS, SOLVE
from my_net import bn10

def make_net():
    pred, c, h = R.r_lstm_step(bn10, num_hidden=3, C=1)
    pred = mx.sym.LogisticRegressionOutput(data=pred, name='softmax')
    sym = mx.symbol.Group([pred, c, h])
    return sym

def train(param=PARAMS, sv=SOLVE, small=False):

    net = make_net()

    out = R.get(2, rate=0.05) 
    train, param['eval_data'] = out['train'], out['val']  
    param['marks'] = param['e_marks'] = out['marks'] 

    s = Solver(net, train, sv, **param)
    s.train()
    s.predict()

if __name__ == '__main__':
    SOLVE['is_rnn'] = True
    SOLVE['load'] = True
    SOLVE['load_perfix'] = 'Result/<26-12:07:31>[E10]/[ACC-0.97690 E9]'
    SOLVE['load_epoch']  = 9

    PARAMS['num_epoch'] = 5
    PARAMS['learning_rate'] = 1e-2
    train()
