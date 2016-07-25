import my_utils as mu
import ipt
import mxnet as mx



PARAMS={
    'ctx':mu.gpu(2),
    'learning_rate':1,
    'num_epoch':10,
    #'optimizer':'adam',
    'initializer':mx.initializer.Xavier(rnd_type='gaussian'),
    #'wd':1,
}

SOLVE = {
    'save_best':True,
    'is_rnn'   :False,  
}



from collections import namedtuple
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias","h2h_weight", "h2h_bias", 'Y_weight', 'Y_bias'])