import ipt
import mxnet as mx
import pickle as pk
import numpy as np
import socket

host = socket.gethostname()
if host == 'NYUSH':
    files = [
                '/home/zijia/HeartDeepLearning/DATA/MADE/[T30,N10]<7-10:04:47>.pk',
                '/home/zijia/HeartDeepLearning/DATA/MADE/[T30,N10]<7-10:04:59>.pk',
                #'/home/zijia/HeartDeepLearning/DATA/MADE/[T30,N10]<7-21:25:32>.pk',
            ] 
else:
    
    files = [
                '/home/zijia/HeartDeepLearning/DATA/PK/NEW/[T30,N10]<6-11:28:45>.pk',
                '/home/zijia/HeartDeepLearning/DATA/PK/NEW/[T30,N10]<6-11:29:01>.pk',
                '/home/zijia/HeartDeepLearning/DATA/PK/NEW/[T30,N10]<26-10:44:34>.pk',
                '/home/zijia/HeartDeepLearning/DATA/PK/NEW/[T30,N10]<26-10:44:43>.pk',
                '/home/zijia/HeartDeepLearning/DATA/PK/NEW/[T30,N10]<26-10:45:03>.pk',
                '/home/zijia/HeartDeepLearning/DATA/PK/NEW/[T30,N10]<26-10:45:25>.pk',
                '/home/zijia/HeartDeepLearning/DATA/PK/NEW/[T30,N10]<26-10:55:08>.pk',
                '/home/zijia/HeartDeepLearning/DATA/PK/NEW/[T30,N10]<26-10:55:20>.pk',
                '/home/zijia/HeartDeepLearning/DATA/PK/NEW/[T30,N10]<26-10:55:48>.pk',
                '/home/zijia/HeartDeepLearning/DATA/PK/NEW/[T30,N10]<26-10:55:59>.pk',
                '/home/zijia/HeartDeepLearning/DATA/PK/NEW/[T30,N10]<26-10:56:37>.pk',
                '/home/zijia/HeartDeepLearning/DATA/PK/NEW/[T30,N10]<26-10:56:53>.pk',
            ]
    f10 = [
            '/home/zijia/HeartDeepLearning/DATA/PK/NEW/[T10,N10]<8-11:42:11>.pk',
            ]

def load_rnn_pk(fnames):
    imgs = None
    labels = None
    for p in fnames:
        with open(p,'r') as f:
            img = pk.load(f)
            ll  = pk.load(f)

        if len(img.shape)==3:
            img = img[None,:,None,:,:] # N*T*1*256*256
            ll  = ll[None,:,None,:,:] 
        elif len(img.shape)==5:
            img = np.transpose(img, axes=(1,0,2,3,4)) # T*N => N*T
            ll = np.transpose(ll, axes=(1,0,2,3,4))

        if imgs is None:
            imgs = img
        else:
            imgs = np.concatenate([imgs,img], axis=0)

        if labels is None:
            labels = ll
        else:
            labels = np.concatenate([labels,ll], axis=0)

    print 'Substract Mean'
    mean = imgs.mean(axis=(2,3,4)).astype(np.uint8)
    i, j = mean.shape
    for x in range(i):
        for y in range(j):
            imgs[x,y] -= mean[x,y]

    return imgs, labels

from HeartDeepLearning.my_utils import prepare_set
from rnn_iter import RnnIter

def create_rnn_iter(img, ll, vimg, vll, batch_size=10, last_batch_handle='pad', num_hidden=250):
    train = RnnIter(img, label=ll, batch_size=batch_size, last_batch_handle=last_batch_handle, num_hidden=num_hidden)

    # N = vimg.shape[1]
    # batch_size = min(N, batch_size)
    val   = RnnIter(vimg,label=vll, batch_size=batch_size, last_batch_handle=last_batch_handle, num_hidden=num_hidden)

    return train, val

def get(bs=1, fs=files, rate=0.1, small=False):
    if small:
        fs = f10
    imgs, labels = load_rnn_pk(fs)

    data = prepare_set(imgs, labels, rate=rate)
    data = list(data)
    for i, a in enumerate(data):
        data[i] = np.transpose(a, axes=(1,0,2,3,4))

    train, val = create_rnn_iter(*data,batch_size=bs)
    T = imgs.shape[1]
    mark = np.ones((T)).astype(np.int)
    return {'train':train, 'val':val, 'marks':mark}

if __name__ == '__main__':
    get(small=True)