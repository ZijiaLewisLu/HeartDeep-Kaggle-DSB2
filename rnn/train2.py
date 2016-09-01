# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import ipt
import numpy as np
import mxnet as mx

from lstm import lstm_unroll
import my_utils as mu
from rnn_iter import get

def Perplexity(label, pred):
    label = label.T.reshape((-1,))
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)

if __name__ == '__main__':
    batch_size = 1
    num_epoch = 25
    small_set = True 


    learning_rate = 0.01
    num_hidden = 4
    num_lstm_layer = 1
    momentum = 0.0

    # dummy data is used to test speed without IO
    dummy_data = False

    contexts = mu.gpu(1)

    def sym_gen(seq_len):
        return lstm_unroll(num_lstm_layer, seq_len, num_hidden=num_hidden, num_label=1)

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden, 256, 256)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden, 256, 256)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    data = get(init_states, bs=batch_size, small=small_set)
    data_train = data['train']
    data_val   = data['val']

    if dummy_data:
        data_train = DummyIter(data_train)
        data_val = DummyIter(data_val)

    num_time = data_train.data_list[0].shape[1]
    symbol = sym_gen(num_time)

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
                                 num_epoch=num_epoch,
                                 learning_rate=learning_rate,
                                 # momentum=momentum,
                                 # wd=0.00001,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    from rnn_metric import rnnM

    model.fit(X=data_train, eval_data=data_val,
              eval_metric=rnnM('test'),
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)

