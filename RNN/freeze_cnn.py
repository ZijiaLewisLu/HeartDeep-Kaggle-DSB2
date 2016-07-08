import ipt
import mxnet as mx
import my_utils as mu
from rnn import rnn_net
from solver import Solver
import rnn_load
from settings import PARAMS, SOLVE
from CNN.cnn import cnn_net
import numpy as np

Good80 = (
    '/home/zijia/HeartDeepLearning/CNN/Result/<0Save>/<1-17:12:45>[E40]/[ACC-0.92900 E39]', 39)


def train(base_model, param=PARAMS, sv=SOLVE, small=False):

    # prepare data
    if small:
        files = rnn_load.f10
        param['ctx'] = mu.gpu(1)
    else:
        files = rnn_load.files

    imgs, labels = rnn_load.load_rnn_pk(files)
    it, lt, iv, lv = mu.prepare_set(imgs, labels)
    N, T = it.shape[:2]

    # cnn process
    model = mx.model.FeedForward.load(*base_model, ctx=mu.gpu(1))
    rnn_input = np.zeros_like(it)
    for n in range(1):
        rnn_input[n], imgs, labels = mu.predict_draw(model, it[n])

    # prepare params
    datas = [rnn_input, lt, iv, lv]
    for i, d in enumerate(datas):
        datas[i] = np.transpose(d,axes=(1,0,2,3,4))
    iters = rnn_load.create_rnn_iter(*datas, batch_size=1)
    param['eval_data'] = iters[1]
    mark = param['marks'] = param['e_marks'] = [1]*T
    rnet = rnn_net(begin=mx.sym.Variable('data'))
    s = Solver(rnet, iters[0], sv, **param)

    # train
    print 'Start Training...'
    s.train()
    s.predict()

if __name__ == '__main__':
    PARAMS['num_epoch'] = 10
    PARAMS['learning_rate'] = 1
    SOLVE['is_rnn'] = True

    train(Good80, small=True)
