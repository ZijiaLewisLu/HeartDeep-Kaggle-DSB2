import ipt
from mxnet.io import NDArrayIter, DataBatch, _init_data
import numpy as np
from mxnet.ndarray import array
from my_layer import LSTM
import mxnet.symbol as S
from settings import LSTMParam



def unroll_lstm(seq_len, num_hidden, C, H, W):
    T = seq_len
    cs = [S.Variable('c')]
    hs = [S.Variable('h')]
    preds  = []
    datas  = [S.Variable('data%d'%i) for i in range(T)]
    param = LSTMParam(  i2h_weight=S.Variable("i2h_weight"),
                        i2h_bias  =S.Variable("i2h_bias"),
                        h2h_weight=S.Variable("h2h_weight"),
                        h2h_bias  =S.Variable("h2h_bias"),
                        Y_weight  =S.Variable("Y_weight"),
                        Y_bias    =S.Variable("Y_bias"))
    for t in range(T):
        pred, c, h = LSTM(datas[t], num_hidden, C, H, W, c=cs[-1], h=hs[-1], param=param)
        pred = S.LogisticRegressionOutput(data=pred, name='logis%d'%t)
        preds.append(pred)
        cs.append(c)
        hs.append(h)
    return S.Group(preds)

class UnrollIter(NDArrayIter):
    def __init__(self, data, label=None, batch_size=1, last_batch_handle='pad', num_hidden=250):
        """data and label should be N*T*C*H*W"""

        self.data  = [] 
        self.label = []
        self.T     = data.shape[1]
        for t in range(self.T):
            self.data += _init_data(data[:,t], allow_empty=False, default_name='data%d'%t)
            label_part = label[:,t] if label is not None else None
            self.label+= _init_data(label_part, allow_empty=True, default_name='logis%d_label'%t)


        self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        self.num_source = len(self.data_list)
        self.num_hidden = num_hidden

        # batching
        if last_batch_handle == 'discard':
            new_n = self.data_list[0].shape[0] - self.data_list[0].shape[0] % batch_size
            data_dict = OrderedDict(self.data)
            label_dict = OrderedDict(self.label)
            for k, _ in self.data:
                data_dict[k] = data_dict[k][:new_n]
            for k, _ in self.label:
                label_dict[k] = label_dict[k][:new_n]
            self.data = data_dict.items()
            self.label = label_dict.items()
        self.num_data = self.data_list[0].shape[0]
        assert self.num_data >= batch_size, \
            "batch_size need to be smaller than data size."
        self.cursor = -batch_size
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        lst = [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.data]

        H, W = self.data_list[0].shape[-2:]

        lst += [
            ('c',(self.batch_size, self.num_hidden)),
            ('h',(self.batch_size, self.num_hidden)),
            ('i2h_weight', (self.num_hidden*4, H*W)),
            ('i2h_bias',   (self.num_hidden*4,)),
            ('h2h_weight', (self.num_hidden*4, H*W)),
            ('h2h_bias',   (self.num_hidden*4,)),
            ('Y_weight', (self.num_hidden, self.num_hidden)),
            ('Y_bias',   (self.num_hidden,)),
            ]
        return lst