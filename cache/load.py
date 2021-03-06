import ipt
# import minpy as mp
#import minpy.numpy as np
import mxnet as mx
import numpy as np
import pickle as pk
import os


def _load_pk_file(fname, rate):
    with open(fname,'r') as f:
        img = pk.load(f)
        ll  = pk.load(f)

    print '-mean'
    img -= img.mean(axis = 0)
    # ll  -= ll.mean (axis = 0)

    img = img[:,None,:,:]
    ll  = ll[:,None,:,:]

    N = img.shape[0]

    p = int(rate * N)

    val_img = img[:p]
    img     = img[p:]
    val_ll  = ll[:p]
    ll      = ll[p:]

    return img, ll, val_img, val_ll

def load_pk(fname, rate = 0.1):
    
    fname = [fname] if not isinstance(fname, list ) else fname

    img_train = ll_train = img_val = ll_val = None 
    for f in fname:
        img, ll , vimg, vll = _load_pk_file(f, rate)

        if img_train == None:
            img_train = img
        else:
            np.concatenate(img_train,img)

        if ll_train == None:
            ll_train = ll
        else:
            np.concatenate(ll_train,ll)

        if img_val == None:
            img_val = vimg
        else:
            np.concatenate(img_val,vimg)

        if ll_val == None:
            ll_val = vll
        else:
            np.concatenate(ll_val,vll)

    # return img_train , ll_train, img_val, ll_val
    # print 'len of val', img_val.shape[0]

    return img_train, ll_train, img_val, ll_val


def create_iter(img,ll,vimg,vll,batch_size =50,last_batch_handle='pad'):

    train = mx.io.NDArrayIter(
            img,
            label=ll, 
            batch_size = batch_size, shuffle=True, last_batch_handle = last_batch_handle)

    # rate = vimg.shape[0]/img.shape[0]
    # print 'val batch size', int(rate * batch_size)

    val   = mx.io.NDArrayIter(
            vimg,
            label=vll, 
            batch_size = batch_size, shuffle = False, last_batch_handle = last_batch_handle)

    return train, val


def load(filename, bs = 10, return_raw = False):
    
    data = load_pk(filename)
    print 'Data Shape, Train %s, Val %s' % ( data[0].shape, data[2].shape )

    train, val = create_iter(**data, bathc_size = bs)

    output = {
        'train': train,
        'val'  : val
        }

    if return_raw:
        output['train_img'] = data[0]
        output['train_label'] = data[1]]
        output['val_img'] = data[2]
        output['val_label'] = data[3]

    return output

def get(bs, small = False, return_raw = False):

    if small:
        f = "/home/zijia/HeartDeepLearning/Net/o1.pk"
    else:
        f = 'home/zijia/HeartDeepLearning/Net/online.pk'

    return load(f,bs = bs, return_raw = return_raw)    



