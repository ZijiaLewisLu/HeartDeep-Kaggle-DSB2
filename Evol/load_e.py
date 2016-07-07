import ipt
import mxnet as mx
import pickle as pk
import numpy as np
import my_utils as mu
from skimage.transform import resize

def get(bs, small=False, aug=False):
    if small:
        filename = "/home/zijia/HeartDeepLearning/DATA/PK/o1.pk"
    else:
        filename = [
            '/home/zijia/HeartDeepLearning/DATA/PK/online.pk',
            '/home/zijia/HeartDeepLearning/DATA/PK/validate.pk',
        ]

    img, label = mu.load_pk(filename)

    L = np.zeros(label.shape[:2]+(64,64))

    for i, ll in enumerate(label):
        ll = ll[0]
        ll = resize(ll,(64,64))
        L[i,0]= ll>5e-4

    print L.mean()

    it, lt, iv, lv = mu.prepare_set(img, L)

    if aug:
        it, lt = mu.augment_sunny(it,lt) 

    print 'Data Shape, Train %s, Val %s' % (it.shape, iv.shape)

    train, val = mu.create_iter(it, lt, iv, lv, batch_size=bs)

    return {'train': train, 'val': val}



if __name__ == '__main__':
    get(1, small=True)