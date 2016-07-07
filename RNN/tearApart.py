import ipt
import mxnet as mx

perfix = './Result/[ACC-0.22647 E9]'
epoch = 9
from rnn import rnn
from rnn_load import get
net = rnn()
iters = get()

from HeartDeepLearning.CNN.cnn_internal import fetch_internal
rout,rimg,rll = fetch_internal(net, iters['val'], perfix, epoch, is_rnn=True)