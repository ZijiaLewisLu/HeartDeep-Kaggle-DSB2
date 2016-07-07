import ipt
import mxnet as mx
import HeartDeepLearning.my_utils as mu 

def fetch_internal(net, val, perfix, epoch):
    
    net = net.get_internals()
    features = [ net[i] for i in range(len(net.list_outputs())) 
                    if net[i].name != 'data' and '_' not in net[i].name] 
    names = [ _.name for _ in features ]
    net = mx.sym.Group(features)
         
    from mxnet.model import load_checkpoint

    sym, arg, aux = load_checkpoint(perfix,epoch)    
    model = mx.model.FeedForward(net, ctx=mu.gpu(1), num_epoch=1, begin_epoch=0)

    shape = dict(val.provide_data+val.provide_label)
    model._init_params(shape)
    model.arg_params.update(arg)
    model.aux_params.update(aux)
    print 'Start Predict'
    outputs = mu.predict_draw(model, iters['val'])
    print len(outputs[0])

    return outputs, names

if __name__ == '__main__':
    perfix = '/home/zijia/HeartDeepLearning/CNN/Result/[ACC-0.93164 E29]' 
    epoch  = 29
    from cnn import cnn_net
    net = cnn_net()
    iters = mu.get(3, small=True)
    fetch_internal(net, iters['val'], perfix, epoch )