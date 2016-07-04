import ipt
import mxnet as mx
import pickle as pk
import numpy as np

fs = [
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:11.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:12.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:13.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:14.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:15.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:16.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:17.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:18.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:19.pk'
]

def load_rnn_pk(fnames):
    imgs = None
    labels = None
    for p in fnames:
        with open(p,'r') as f:
            img = pk.load(f)
            ll  = pk.load(f)

        img = img[None,:,None,:,:]
        ll  = ll[None,:,None,:,:]

        if imgs is None:
            imgs = img
        else:
            imgs = np.concatenate([imgs,img], axis=0)

        if labels is None:
            labels = ll
        else:
            labels = np.concatenate([labels,ll], axis=0)

    return imgs, labels

from HeartDeepLearning.my_utils import prepare_set
from rnn_iter import RnnIter

def create_cnn_iter(img, ll, vimg, vll, batch_size=10, last_batch_handle='pad'):
    train = RnnIter(img, label=ll, batch_size=batch_size, last_batch_handle=last_batch_handle)

    N = vimg.shape[0]
    batch_size = min(N, batch_size)
    val   = RnnIter(vimg,label=vll, batch_size=batch_size, last_batch_handle=last_batch_handle)

    return train, val

def get(bs=1):
    imgs, labels = load_rnn_pk(fs)
    it, lt, iv, lv = prepare_set(imgs, labels)
    train, val = create_cnn_iter(it,lt,iv,lv,batch_size=bs)
    mark = np.ones((30))
    return {'train':train, 'val':val, 'marks':mark}

if __name__ == '__main__':
    get()