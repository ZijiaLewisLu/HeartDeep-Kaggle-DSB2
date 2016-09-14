import ipt
from mxnet.io import DataIter, DataBatch, _init_data
import numpy as np
from mxnet.ndarray import array


class RIter(DataIter):
    def __init__(self, data, init_status, label=None, batch_size=1, last_batch_handle='pad'):
        """data and label should be T*N*C*H*W"""
        super(RIter, self).__init__()

        self.data = _init_data(data, allow_empty=False, default_name='data')
        self.label = _init_data(label, allow_empty=True, default_name='softmax_label')

        self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        self.num_source = len(self.data_list)

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
        # self.num_t = self.data_list[0].shape[1]
        self.init_status = init_status

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        lst = [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.data]
        label_shape = [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.label]
        return lst + self.init_status + label_shape

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return []


    def hard_reset(self):
        """Igore roll over data and set to start"""
        self.cursor = -self.batch_size

    def reset(self):
        if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
            self.cursor = -self.batch_size + (self.cursor%self.num_data)%self.batch_size
        else:
            self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            return self.get_next()
        else:
            raise StopIteration

    def get_next(self):
        """in zoo, each DataBatch has pic of one time step of sample of #batch_size """
        assert(self.cursor < self.num_data), "DataIter needs reset."


        if self.cursor + self.batch_size <= self.num_data:
            data = [ array(d[1][self.cursor:self.cursor+self.batch_size])
                            for d in self.data]
            label = [ array(l[1][self.cursor:self.cursor+self.batch_size])
                            for l in self.label]
        else:
            pad = self.batch_size - self.num_data + self.cursor
            data = [array(np.concatenate(
                        (d[1][self.cursor:], d[1][:pad]), axis=0)) for d in self.data]
            label = [array(np.concatenate(
                        (l[1][self.cursor:], l[1][:pad]), axis=0)) for l in self.label]

        batch = DataBatch(data = data, label = label, pad= self.getpad(), index = None)
            
        return batch

    def getpad(self):
        if self.last_batch_handle == 'pad' and \
           self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0


    
def create_iter(img, ll, vimg, vll, init_status,  batch_size=10, last_batch_handle='pad'):
    train = RIter(img,  init_status, label=ll, batch_size=batch_size, last_batch_handle=last_batch_handle)
    val   = RIter(vimg, init_status, label=vll, batch_size=batch_size, last_batch_handle=last_batch_handle)
    return train, val

def get(init_status, bs=1, fs=None, rate=0.1, small=False):
    from RNN.rnn_load import files, f10, load_rnn_pk
    from my_utils import prepare_set
    import numpy as np

    if small:
        fs = f10
    elif fs is None:
        fs = files

    imgs, labels = load_rnn_pk(fs)
    print 'IMAGE SHAPE', imgs.shape, labels.shape
    data = prepare_set(imgs, labels, rate=rate)
    data = list(data)


    data.append(init_status)
    train, val = create_iter(*data, batch_size=bs)
    return {'train':train, 'val':val}
