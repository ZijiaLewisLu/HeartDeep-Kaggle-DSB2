import ipt
import minpy as mp
import mxnet as mx
import minpy.numpy as np

import net as n, load_data as load

PARAMS = n.Params
PK = 'o1.pk' 


def train(model,dataiter,call_back=None):
    model.fit(
            dataiter,
            batch_end_callback = call_back
            )


if __name__ is '__main__':
    net = n.create_net(PARAMS)
    model = n.create_mxnet(net)

    img,ll = load.load_pk(PK)
    diter = mx.io.NDArrayIter(img, ll, batch_size = 50,last_batch_handle = 'pad')

    print 'start to train'
    train(model, diter, 
            call_back = n.batch_call_back
            )
    
