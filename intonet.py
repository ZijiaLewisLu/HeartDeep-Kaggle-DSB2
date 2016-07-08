import ipt
import mxnet as mx
import HeartDeepLearning.my_utils as mu
from collections import OrderedDict
import math


def show_layer(layers, *args):
    if isinstance(layers, str):
        layers = [layers]
    N = len(args)
    MAX_COL = 4
    row = math.ceil(N / MAX_COL)
    fig, subs = plt.subplots(row, MAX_COL)
    for layer in layers:
        print '-' * 30, layer
        for idx, output in enumerate(args):
            L = len(output[layer].shape)
            if L == 4:
                subs[idx].imshow(output[layer][0, 0], cmap='gray')
            elif L == 2:
                subs[idx].imshow(output[layer], cmap='gray')
            else:
                print 'Abandoned', idx, output[layer].shape
        plt.show()
        fig.clear()
    plt.close()


def show_filter(layer, *args):
    print layer
    N = len(args)
    MAX_COL = 4
    row = math.ceil(N / MAX_COL)
    fig, subs = plt.subplots(row, MAX_COL)
    shape = args[0][layer].shape
    H = shape[1]
    L = len(shape)
    if L == 2:
        for idx, output in enumerate(args):
            print '_' * 30, h
            subs[idx].imshow(rout[layer][0, h], cmap='gray')
            plt.show()
        fig.clear()
        plt.close()
        return

    for h in range(H):
        for idx, output in enumerate(args):
            print '_' * 30, h
                subs[idx].imshow(rout[layer][0, h], cmap='gray')
        plt.show()
        fig.clear()
    plt.close()


def fetch_internal(net, val, perfix, epoch, is_rnn=False):

    def verify(name):

        for _ in ['weight', 'bias', 'gamma', 'beta', 'blockgrad', 'data', 'label']:
            if _ in name:
                print 'Abandoned:', name
                return False
        # if name.startswith('_'):
        #    return False
        for _ in ['c', 'h']:
            if name == _:
                print 'Abandoned:', name
                return False
        return True

    net = net.get_internals()
    print '\n', net.list_outputs(), '\n'
    features = [net[i]
                for i in range(len(net.list_outputs())) if verify(net[i].name)]
    names = [_.name for _ in features]
    net = mx.sym.Group(features)

    from mxnet.model import load_checkpoint

    sym, arg, aux = load_checkpoint(perfix, epoch)
    if not is_rnn:
        model = mx.model.FeedForward(
            net, ctx=mu.gpu(1), num_epoch=1, begin_epoch=0)
    else:
        from HeartDeepLearning.RNN import rnn_feed
        model = rnn_feed.Feed(net, ctx=mu.gpu(1), num_epoch=1, begin_epoch=0)

    shape = OrderedDict(val.provide_data + val.provide_label)
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