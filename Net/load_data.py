import ipt
import minpy as mp
#import minpy.numpy as np
import mxnet as mx
import numpy as np
import pickle as pk


def _load_pk_file(fname, rate):
    with open(fname,'r') as f:
        img = pk.load(f)
        ll  = pk.load(f)

    img = img[:,None,:,:]
    ll  = ll[:,None,:,:]

    N = img.shape[0]

    val_img = img[:rate*N]
    img     = img[rate*N:]
    val_ll  = ll[:rate*N]
    ll      = ll[rate*N:]

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

    return img_train, ll_train, img_val, ll_val


def create_iter(img,ll,vimg,vll,batch_size =50,last_batch_handle='pad'):

    train = mx.io.MNISTIter(img, ll, 
            batch_size = batch_size, shuffle=True, last_batch_handle = last_batch_handle)

    rate = vimg.shape[0]/img.shape[0]
    val   = mx.io.MNISTIter(vimg, ll, 
            batch_size = batch_size*rate, shuffle = False, last_batch_handle = last_batch_handle)

    return train, val

    


