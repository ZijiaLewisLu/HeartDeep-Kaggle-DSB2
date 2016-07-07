import my_utils as mu
import ipt
import mxnet as mx
PARAMS={
    'ctx':mu.gpu(2),
    'learning_rate':3,
    'num_epoch':15,
    #'optimizer':'adam',
    'initializer':mx.initializer.Xavier(rnd_type='gaussian'),
    #'wd':1,
}

SOLVE = {
    'save_best':True,
    'is_rnn'   :False,  
}