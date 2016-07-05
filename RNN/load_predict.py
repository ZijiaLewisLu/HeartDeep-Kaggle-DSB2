import ipt
import mxnet as mx
import my_utils as mu
import logging
import rnn
from rnn_load import get
from HeartDeepLearning.solver import Solver

Slow200  = ('/home/zijia/HeartDeepLearning/RNN/Result/<0Save>/<4-22:38:54>TEST[E200]/[ACC-0.06555 E199]', 199)

PARAMS={
    'ctx':mu.gpu(2),
    'learning_rate':6,
    'num_epoch':15,
    'initializer':mx.initializer.Xavier(rnd_type='gaussian'),
}

SOLVE = {
    'save_best':True,
    'is_rnn'   :True,  
}


def predict(param = PARAMS, sv=SOLVE, small=False):
    sv['load'] = True
    sv['load_perfix'], sv['load_epoch'] = Slow200

    sv['name'] = 'Pred'
    net = rnn.rnn()
    out = get(1, rate=0.1)
    train, param['eval_data'] = out['train'], out['val']  
    param['marks'] = param['e_marks'] = out['marks'] 
    s = Solver(net, train, sv, **param)
    s.predict()

if __name__ == '__main__':
    predict()