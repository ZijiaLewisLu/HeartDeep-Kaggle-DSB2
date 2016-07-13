import ipt
import mxnet as mx
import pickle as pk
import numpy as np
import my_utils as mu
from skimage.transform import resize

def reshape_label(lls, scale=4):
    if not isinstance(lls, list):
        lls = [lls]

    for idx, ll in enumerate(lls):
        # prepare shape
        heads = ll.shape[:-2]
        H, W = ll.shape[-2:]
        ll = ll.reshape((-1,H,W))    
        H = H/scale
        W = W/scale

        # start resize
        out = np.empty((ll.shape[0],H,W))
        for i, l in enumerate(ll):
            out[i] = resize(l,(H,W))

        # reshape back
        out = out.reshape(heads+(H,W))
        lls[idx] = out>5e-4

    lls = lls[0] if len(lls)==1 else lls
    return lls


def get(bs, small=False, aug=False):
    if small:
        filename = "/home/zijia/HeartDeepLearning/DATA/PK/o1.pk"
    else:
        filename = [
            '/home/zijia/HeartDeepLearning/DATA/PK/online.pk',
            '/home/zijia/HeartDeepLearning/DATA/PK/validate.pk',
        ]

    img, label = mu.load_pk(filename)
    it, lt, iv, lv = mu.prepare_set(img, label)

    if aug:
        it, lt = mu.augment_sunny(it,lt) 

    Lt, Lv = reshape_label([lt,lv])
    
    print 'Data Shape, Train %s, Val %s' % (it.shape, iv.shape)

    train, val = mu.create_iter(it, Lt, iv, Lv, batch_size=bs)

    return {'train': train, 'val': val}



def get_rnn(bs, small=False, aug=False, rate=0.1):
    import RNN.rnn_load as r 

    fs = r.f10 if small else r.files
    imgs, labels = r.load_rnn_pk(fs)
    data = mu.prepare_set(imgs, labels, rate=rate)
    data = list(data)
    data[1], data[3] = reshape_label(data[1::2]) 
    for i, a in enumerate(data):
        data[i] = np.transpose(a, axes=(1,0,2,3,4))

    hidden = data[1].shape[-1]**2

    train, val = r.create_rnn_iter(*data,batch_size=bs, num_hidden=hidden)
    mark = [1]*imgs.shape[0]
    return {'train':train, 'val':val, 'marks':mark}

if __name__ == '__main__':
    get_rnn(1, small=True)