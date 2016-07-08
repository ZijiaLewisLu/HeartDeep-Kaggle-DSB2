import ipt
import mxnet as mx
import HeartDeepLearning.my_utils as mu 
from collections import OrderedDict

def fetch_internal(net, val, perfix, epoch, is_rnn=False):
    
    def verify(name):

        for _ in ['weight','bias','gamma','beta','blockgrad','data', 'label']:
            if _ in name:
                print 'Abandoned:',name
                return False
        #if name.startswith('_'):
        #    return False
        for _ in ['c','h']:
            if name==_:
                print 'Abandoned:',name
                return False
        return True
    
    net = net.get_internals()
    print '\n', net.list_outputs(), '\n'
    features = [ net[i] for i in range(len(net.list_outputs())) if verify(net[i].name) ]
    names = [ _.name for _ in features ]
    net = mx.sym.Group(features)
         
    from mxnet.model import load_checkpoint

    sym, arg, aux = load_checkpoint(perfix,epoch)    
    if not is_rnn:
        model = mx.model.FeedForward(net, ctx=mu.gpu(1), num_epoch=1, begin_epoch=0)
    else:
        from HeartDeepLearning.RNN import rnn_feed
        model = rnn_feed.Feed(net, ctx=mu.gpu(1), num_epoch=1, begin_epoch=0)
    
    shape = OrderedDict(val.provide_data+val.provide_label)
    model._init_params(shape)
    model.arg_params.update(arg)
    model.aux_params.update(aux)
    print 'Start Predict'
    outputs, img, label = mu.predict_draw(model, val)
    outputs = dict(zip(names, outputs))
    print 'Done'
    return outputs, img, label

if __name__ == '__main__':
    perfix = '/home/zijia/HeartDeepLearning/CNN/Result/[ACC-0.93164 E29]' 
    epoch  = 29
    from CNN.cnn import cnn_net
    net = cnn_net()
    iters = mu.get(3, small=True)
    fetch_internal(net, iters['val'], perfix, epoch )
