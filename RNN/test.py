import numpy as np
from rnn_metric import RnnM
import ipt
import mxnet as mx
from rnn_feed import Feed
from HeartDeepLearning.my_utils import *
import matplotlib.pyplot as plt
from rnn_iter import RnnIter
import matplotlib.image as mpimg

def simple_test():
    a = np.ones((2,2))
    b = np.ones((2,2))
    b[0,:] = 0

    from utils import eval_iou

    print eval_iou(a,b)

    metric = RnnM(eval_iou)

    a = mx.ndarray.array(a)
    b = mx.ndarray.array(b)
    metric.update(a,b)
    print metric.sum_metric, metric.num_inst
    print metric.get()
    metric.update(a,b)
    print metric.get()

    c = np.zeros((2,2))
    c[0,0] = 1
    eval_iou(a.asnumpy(),c)

    c = mx.nd.array(c)
    metric.update(a,c)
    print metric.get()

def get_iter():
    from utils import get
    data = get(1, small = True)
    img = data['train'].data[0][1]
    label = data['train'].label[0][1]

    img = img[0].reshape((1,1,1,256,256))
    label = label[0].reshape((1,1,1,256,256))

    return RnnIter(img,label)


def test_net():
    model = Feed.load('12_epoch',12)


    model.num_epoch = 1
    
    # N = img.shape[0]
    # for i in range(N):
    # pred = model.simple_pred(img[0])
    c = Callback()
    model.fit(
        get_iter(),
        [1],
        eval_metric = RnnM(eval_iou),
        epoch_end_callback = c.epoch,
        batch_end_callback = c.batch,
        )
    # plt.imshow(pred[0,0])
    # plt.savefig('pred.png')
    # plt.close()

    # plt.imshow(label[0,0])
    # plt.savefig('label.png')
    # plt.close()

def test_iou():
    img = mpimg.imread('pred.png')
    label = mpimg.imread('label.png')
    print img.shape

def test_callback():
    c = Callback()

    # print c.epoch()
    print c.eval
    print c.reset
    print type(c.epoch), c.epoch,
    print type(c.batch), c.batch


def test_iter_bs():
    from rnn_load import get
    output = get(bs=2)
    train = output['train']
    for zoo in train:
        print '___________'
        for databatch in zoo:
            print databatch.data[0].shape

    train.reset()
    print 'reset'
    for zoo in train:
        print '___________'
        for databatch in zoo:
            print databatch.data[0].shape

    

    
if __name__ == '__main__':
    test_iter_bs()
    # test_net()