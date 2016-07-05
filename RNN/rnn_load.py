import ipt
import mxnet as mx
import pickle as pk
import numpy as np

files = [
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:11.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:12.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:13.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:14.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:15.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:16.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:17.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:18.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]30-23:49:19.pk',
'/home/zijia/HeartDeepLearning/DATA/PK/New_DATA/[30]-4.pk',
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

    # N = vimg.shape[1]
    # batch_size = min(N, batch_size)
    val   = RnnIter(vimg,label=vll, batch_size=batch_size, last_batch_handle=last_batch_handle)

    return train, val

def get(bs=1, fs=files, rate=0.1, small=False):
    if small:
        fs = fs[:1]
    imgs, labels = load_rnn_pk(fs)
    data = prepare_set(imgs, labels, rate=rate)
    data = list(data)
    for i, a in enumerate(data):
        data[i] = np.transpose(a, axes=(1,0,2,3,4))

    train, val = create_cnn_iter(*data,batch_size=bs)
    mark = np.ones((30)).astype(np.int)
    return {'train':train, 'val':val, 'marks':mark}

if __name__ == '__main__':
    get()