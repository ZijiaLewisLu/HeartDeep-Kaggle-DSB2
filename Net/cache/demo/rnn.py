import ipt
import net as n
import mxnet as mx
import numpy as np
def net():
    DATA = mx.sym.Variable(name = 'data')
    C   = mx.sym.Variable(name = 'c')
    H    = DATA + C

    PRED = mx.sym.FullyConnected(name='fc', data = H, num_hidden = 10)

    return mx.symbol.Group([PRED,C])

def run():
    G = net()
    d = mx.random.normal(0,1,shape=(20,10))
    c = mx.random.normal(0,3,shape=(20,10))
    weight = mx.random.normal(0,1,shape=(10,10))
    bias = mx.random.normal(0,1,shape=(10,))
    print G.infer_shape(data=(20,10), c = (20,10))
    print G.list_arguments()
    gd = mx.nd.empty((20,10))

    exe = G.bind(ctx = mx.context.gpu(0), args= {'data': d, 'c': c, 'fc_weight':weight, 'fc_bias':bias}, args_grad={'data':gd}) 
   
    exe.forward()

if __name__ == '__main__':
    run()