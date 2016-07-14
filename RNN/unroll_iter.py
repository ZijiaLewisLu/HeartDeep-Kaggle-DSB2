from mxnet.io import NDArrayIter, DataBatch, _init_data
import numpy as np
from mxnet.ndarray import array


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
        lst.append(('c',(self.batch_size, self.num_hidden)))
        lst.append(('h',(self.batch_size, self.num_hidden)))
        return lst