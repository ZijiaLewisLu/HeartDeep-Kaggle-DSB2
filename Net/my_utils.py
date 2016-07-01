import ipt
import time
import os
import logging
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
import matplotlib.pyplot as plt
import os
import pickle as pk
from PIL import Image
import copy
import GPU_availability as G

def gpu(num):
    gs = G.GPU_availability()[:num]
    return [ mx.context.gpu(g) for g in gs ]

def predict_draw(model, val, folder=None):
    if model.arg_params is None:
        print 'INIT MODEL...'
        d = val.provide_data
        l = val.provide_label
        model._init_params(dict(d + l))

    out = model.predict(
            val,
            # num_batch=4,
            return_data=True
        )

    if folder is None:
        folder = parse_time() + 'Prediction'
    
    if not folder.endswith('/'):
        folder += '/'

    try:
        os.mkdir(folder)
    except OSError, e:
        print e, 'ecountered'

    N = out[0].shape[0]

    for idx in range(N):
        gap = np.ones((256,5))
        pred= out[0][idx,0]
        img = out[1][idx,0]
        label = out[2][idx,0]
        png = np.hstack([pred,gap,label])
        
        print 'Pred mean>>',pred.mean(), 'Label mean>>', label.mean()

        fig = plt.figure()
        fig.add_subplot(121).imshow(png)
        fig.add_subplot(122).imshow(img)
        fig.savefig(folder+'Pred%d.png'%(idx))
        fig.clear()
        plt.close('all')

def parse_time():
    now = time.ctime(int(time.time()))
    now = now.split(' ')
    return '<'+now[-3]+'-'+now[-2]+'>'


def plot_save(img, name):
    plt.imshow(img)
    if 'png' not in name:
        name = name + '.png'
    plt.savefig(name)
    plt.close()


def eval_iou(label, pred):

    conjunct = pred * label
    union = pred + label
    out = np.sum(conjunct * 2) / np.sum(union)

    if not 0 <= out <= 1:
        print 'eval error >>', out, np.sum(conjunct), np.sum(union)

    return out

###################################################
# Load Data
def _load_pk_file(fname, rate):
    with open(fname, 'r') as f:
        img = pk.load(f)
        ll = pk.load(f)

    #print '-mean'
    img -= img.mean(axis=0)
    # ll  -= ll.mean (axis = 0)

    img = img[:, None, :, :]
    ll = ll[:, None, :, :]

    N = img.shape[0]

    p = int(rate * N)

    val_img = img[:p]
    img = img[p:]
    val_ll = ll[:p]
    ll = ll[p:]

    return img, ll, val_img, val_ll


def load_pk(fname, rate=0.1):

    fname = [fname] if not isinstance(fname, list) else fname

    img_train = ll_train = img_val = ll_val = None
    for f in fname:
        img, ll, vimg, vll = _load_pk_file(f, rate)

        if img_train == None:
            img_train = img
        else:
            img_train = np.concatenate((img_train, img))

        if ll_train == None:
            ll_train = ll
        else:
            ll_train = np.concatenate((ll_train, ll))

        if img_val == None:
            img_val = vimg
        else:
            img_val = np.concatenate((img_val, vimg))

        if ll_val == None:
            ll_val = vll
        else:
            ll_val = np.concatenate((ll_val, vll))

    # return img_train , ll_train, img_val, ll_val
    # print 'len of val', img_val.shape[0]

    return img_train, ll_train, img_val, ll_val


def create_iter(img, ll, vimg, vll, batch_size=50, last_batch_handle='pad'):

    train = mx.io.NDArrayIter(
        img,
        label=ll,
        batch_size=batch_size, shuffle=True, last_batch_handle=last_batch_handle)

    # rate = vimg.shape[0]/img.shape[0]
    # print 'val batch size', int(rate * batch_size)

    val = mx.io.NDArrayIter(
        vimg,
        label=vll,
        batch_size=batch_size, shuffle=False, last_batch_handle=last_batch_handle)

    return train, val


def load(filename, bs=10, return_raw=False):

    data = load_pk(filename)
    print 'Data Shape, Train %s, Val %s' % (data[0].shape, data[2].shape)

    train, val = create_iter(*data, batch_size=bs)

    output = {
        'train': train,
        'val': val
    }

    if return_raw:
        output['train_img'] = data[0]
        output['train_label'] = data[1]
        output['val_img'] = data[2]
        output['val_label'] = data[3]

    return output


def get(bs, small=False, return_raw=False):

    if small:
        f = "/home/zijia/HeartDeepLearning/Net/data/o1.pk"
    else:
        f = [
            '/home/zijia/HeartDeepLearning/Net/data/online.pk',
            '/home/zijia/HeartDeepLearning/Net/data/validate.pk',
        ]

    return load(f, bs=bs, return_raw=return_raw)





