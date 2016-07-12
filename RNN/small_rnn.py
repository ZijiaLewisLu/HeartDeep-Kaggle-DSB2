import ipt
import mxnet as mx
import my_utils as mu
from rnn import rnn_net
from solver import Solver
import rnn_load
from settings import PARAMS, SOLVE
import numpy as np
from my_layer import LSTM


def warp(func):
    def part(label, preds):
        pred = preds[0]
        label = mx.nd.array(label)
        return func([label], [pred])
    return part

def train(param=PARAMS, sv=SOLVE, small=False):
    # prepare net
    data = mx.sym.Variable(name='data')
    pred, c, h = LSTM(data, 125, 1, 50, 50)
    logis = mx.sym.LogisticRegressionOutput(data=pred, name='softmax')
    group = mx.sym.Group([logis, c, h])

    # prepare data
    img = np.random.randn(10,10,1,50,50)
    ll  = img/2
    datas = mu.prepare_set(img, ll)
    datas = list(datas)
    N, T = datas[0].shape[:2]

    # prepare params
    for i, d in enumerate(datas):
        datas[i] = np.transpose(d,axes=(1,0,2,3,4))
        # make T become one
        #datas[i] = d.reshape((-1,1)+d.shape[2:])

    iters = rnn_load.create_rnn_iter(*datas, batch_size=1, num_hidden=125)
    param['eval_data'] = iters[1]
    mark = param['marks'] = param['e_marks'] = [1]*T
    s = Solver(group, iters[0], sv, **param)

    # train
    print 'Start Training...'
    s.train()
    s.predict()

if __name__ == '__main__':
    PARAMS['num_epoch'] = 1000
    PARAMS['learning_rate'] = 10
    PARAMS['ctx'] = mu.gpu(1)
    

    SOLVE['load'] = False
    SOLVE['load_perfix'] = '/home/zijia/HeartDeepLearning/RNN/Result/<9-10:28:52>LSTM[E50]/[ACC-0.34549 E49]'
    SOLVE['load_epoch']  = 49

    SOLVE['is_rnn'] = True
    SOLVE['name'] = __file__

    train(small=True)
