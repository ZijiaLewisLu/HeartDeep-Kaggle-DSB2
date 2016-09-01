import ipt
import mxnet as mx
import my_utils as mu
import os

pks = os.listdir('../DATA/PK/patience')
files = [ os.path.join('../DATA/PK/patience', _)  for _ in pks if _.endswith('.pk') ]
print files

from RNN.rnn_load import load_rnn_pk

images = []
labels = []
marks = []
import pickle, numpy as np
MAX = 0
for f in files:
    with open(f,'r') as pk:
        images.append(pickle.load(pk))
        labels.append(pickle.load(pk))

        m = pickle.load(pk)
        if len(m)>MAX:
            MAX = len(m)
        marks.append(m)


num_patience = len(images)
for i in range(num_patience):
    T = len(marks[i])
    if T < MAX:
        res = MAX-T
        makeup = np.zeros((res,256,256))
        images[i] = np.concatenate((images[i], makeup))
        labels[i] = np.concatenate((labels[i], makeup))
        marks[i]  = marks[i] + [0] * res
    images[i] = images[i].reshape((-1,1,1,256,256))
    labels[i] = labels[i].reshape((-1,1,1,256,256))

split = max(1, num_patience*0.1)
timg = np.concatenate(images[:-split], axis=1)
tll  = np.concatenate(labels[:-split], axis=1)
print timg.shape, tll.shape
vimg = np.concatenate(images[-split:], axis=1)
vll  = np.concatenate(labels[-split:], axis=1)


from r_lstm import R_LSTM_Iter
train = R_LSTM_Iter(timg, label=tll, num_hidden=3, batch_size=1)
val =   R_LSTM_Iter(vimg, label=vll, num_hidden=3, batch_size=1)

from solver import Solver
from train import make_net
from settings import PARAMS, SOLVE

SOLVE['is_rnn'] = True
SOLVE['load']   = True
SOLVE['load_perfix'] = '/home/zijia/HeartDeepLearning/R_LSTM/Result/<26-12:43:22>[E5]/[ACC-0.97747 E4]'
SOLVE['load_epoch']  = 4
 
PARAMS['eval_data'] = val 
PARAMS['marks'] = marks[:-split]
PARAMS['e_marks'] = marks[-split:]
PARAMS['ctx'] = mu.gpu(1)

PARAMS['learning_rate'] = 1

s = Solver(make_net(), train, SOLVE, **PARAMS)

s.train()
s.predict()
