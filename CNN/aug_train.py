import ipt
from my_utils import *
from my_layer import *
import mxnet as mx
import cPickle as pk
import HeartDeepLearning.image_transform as tf
import numpy as np

def get():
    fname = [
        '/home/zijia/HeartDeepLearning/Net/data/online.pk',
        '/home/zijia/HeartDeepLearning/Net/data/validate.pk'
    ]
    
    it, lt, iv, lv = load_pk(fname)  
    it = it.reshape(-1,256,256)
    lt = lt.reshape(-1,256,256)
    
    N = it.shape[0]
    print N

    aug = tf.NO_AUGMENT_PARAMS

    #it_end = np.array(it)
    #lt_end = np.array(lt)
    x, y = it[0].shape

    end_i = np.zeros((N*5,x,y))
    end_l = np.zeros((N*5,x,y))

    for idx in range(N):
        img = it[idx]
        ll  = lt[idx]
        for i in range(5):
            zx, zy, sx, sy = np.random.randint(80,high=120,size=4)/100.00
            aug['zoom_x'] = zx
            aug['zoom_y'] = zy
            aug['skew_x'] = sx
            aug['skew_y'] = sy
            aug['rotate'] = np.random.randint(360)
            aug['translate_x'], aug['translate_y'] = np.random.randint(50,size=2)

            i_aug = tf.resize_and_augment_sunny(img, augment=aug)
            l_aug = tf.resize_and_augment_sunny(ll, augment=aug)
            l_aug = (l_aug>0.4)

            end_i[idx*5+i]=i_aug
            end_l[idx*5+i]=l_aug

    end_i = np.concatenate((end_i, it),axis=0)
    end_l = np.concatenate((end_l, lt),axis=0)

    print end_i.shape
    for m in end_i.mean(axis=(1,2)):
        print m
    assert end_i.mean(axis=(1,2)).all()!=0

    print 'start dump'
    with open('/home/zijia/HeartDeepLearning/Net/data/aug_all.pk','w') as f:
        pk.dump(end_i,f)
        pk.dump(end_l,f)


if __name__ == '__main__':
    get()




