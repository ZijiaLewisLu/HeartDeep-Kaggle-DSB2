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
import HeartDeepLearning.DATA.image_transform as tf
import numpy as np
import matplotlib.pyplot as plt


def show(*imgs, **kw):
    func = kw.pop('f', 'imshow')
    L = len(imgs)
    if not isinstance(func, list):
        func = [func] * L
    fig, subs = plt.subplots(1, L)
    if L == 1:
        subs = [subs]
    for i in range(L):
        kwords = {}
        if func[i] == 'imshow':
            subs[i].__getattribute__(func[i])(imgs[i], cmap='gray')
        elif func[i] == 'plot':
            subs[i].__getattribute__(func[i])(imgs[i], 'o')

    plt.show()
    fig.clear()
    plt.close()

# Credit to Gaiyu


def GPU_availability():
    import itertools
    from subprocess import Popen, PIPE
    import re
    output = Popen(['nvidia-smi'], stdout=PIPE).communicate()[0]
    lines = output.split('\n')
    performance = {}
    index = 0
    for i in range(len(lines)):
        if 'GTX' in lines[i]:
            p = int(re.search(r'P(\d?\d)', lines[i + 1]).group(0)[-1])
            if p > 1:
                try:
                    performance[p].append(index)
                except:
                    performance.update({p: [index]})
            index += 1
    return list(itertools.chain(*[performance[key] for key in reversed(sorted(performance.keys()))]))


def gpu(num):
    gs = GPU_availability()[:num]
    return [mx.context.gpu(g) for g in gs]


def predict_draw(model, val, draw=False, folder=None):
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

    if not draw:
        return out

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
        gap = np.ones((256, 5))
        pred = out[0][idx, 0]
        img = out[1][idx, 0]
        label = out[2][idx, 0]
        png = np.hstack([pred, gap, label])

        print 'Pred mean>>', pred.mean(), 'Label mean>>', label.mean()

        fig = plt.figure()
        fig.add_subplot(121).imshow(png)
        fig.add_subplot(122).imshow(img)
        fig.savefig(folder + 'Pred%d.png' % (idx))
        fig.clear()
        plt.close('all')


def parse_time():
    now = time.ctime(int(time.time()))
    now = now.split(' ')
    return '<' + now[-3] + '-' + now[-2] + '>'


def save_img(img, name):
    plt.imshow(img)
    if 'png' not in name:
        name = name + '.png'
    plt.savefig(name)
    plt.close()


def eval_iou(label, pred):

    print 'in'

    conjunct = pred * label
    union = pred + label
    out = np.sum(conjunct * 2) / np.sum(union)

    if not 0 <= out <= 1:
        print 'eval error >>', out, np.sum(conjunct), np.sum(union)

    return out

###################################################
# Load Data


def load_pk(fname):
    fname = [fname] if not isinstance(fname, list) else fname

    img_all = None
    label_all = None

    for single_f in fname:
        with open(single_f, 'r') as f:
            img = pk.load(f)
            ll = pk.load(f)

        # print '-mean'
        mean = img.mean(axis=(1, 2)).astype(np.int64).reshape((-1,))
        for i in range(len(mean)):
            img[i] -= mean[i]

        if img_all is None:
            img_all = img
        else:
            img_all = np.concatenate((img_all, img))

        if label_all is None:
            label_all = ll
        else:
            label_all = np.concatenate((label_all, ll))

    img_all = img_all[:, None, :, :]
    label_all = label_all[:, None, :, :]

    return img_all, label_all


def prepare_set(img, label, rate=0.1):

    N = img.shape[0]

    split = max(1, int(round(N * rate)))

    img_train, ll_train = img[split:], label[split:]
    img_val, ll_val = img[:split], label[:split]

    return img_train, ll_train, img_val, ll_val


def create_iter(img, ll, vimg, vll, batch_size=10, last_batch_handle='pad', shuffle=True):

    train = mx.io.NDArrayIter(
        img,
        label=ll,
        batch_size=batch_size, shuffle=shuffle, last_batch_handle=last_batch_handle)

    # rate = vimg.shape[0]/img.shape[0]
    # print 'val batch size', int(rate * batch_size)

    val = mx.io.NDArrayIter(
        vimg,
        label=vll,
        batch_size=batch_size, shuffle=shuffle, last_batch_handle=last_batch_handle)

    return train, val


def augment_sunny(img, label):
    """input has the shape of N*1*256*256"""
    aug = tf.NO_AUGMENT_PARAMS

    N = img.shape[0]
    x, y = img[0, 0].shape
    
    end_i = np.zeros((N * 5, 1, x, y))
    end_l = np.zeros((N * 5, 1, x, y))

    for idx in range(N):
        i = img[idx, 0]
        ll = label[idx, 0]
        for j in range(5):
            zx, zy, sx, sy = np.random.randint(80, high=120, size=4) / 100.00
            aug['zoom_x'] = zx
            aug['zoom_y'] = zy
            aug['skew_x'] = sx * 15
            aug['skew_y'] = sy * 15
            aug['rotate'] = np.random.randint(360)
            aug['translate_x'], aug[
                'translate_y'] = np.random.randint(-50, 50, size=2)

            i_aug = tf.resize_and_augment_sunny(i, augment=aug)
            l_aug = tf.resize_and_augment_sunny(ll, augment=aug)
            l_aug = (l_aug > 0.4)

            end_i[idx * 5 + j, 0] = i_aug
            end_l[idx * 5 + j, 0] = l_aug

    end_i = np.concatenate((end_i, img), axis=0)
    end_l = np.concatenate((end_l, label), axis=0)

    where = end_i.mean(axis=(1,2,3))
    where = (where==0)

    assert end_i.mean(axis=(1, 2, 3)).all() != 0 
    return end_i, end_l


def get(bs, small=False, return_raw=False, aug=False):

    if small:
        filename = "/home/zijia/HeartDeepLearning/DATA/PK/o1.pk"
    else:
        filename = [
            '/home/zijia/HeartDeepLearning/DATA/PK/online.pk',
            '/home/zijia/HeartDeepLearning/DATA/PK/validate.pk',
        ]

    img, label = load_pk(filename)
    it, lt, iv, lv = prepare_set(img, label)

    if aug:
        it, lt = augment_sunny(it, lt)

    print 'Data Shape, Train %s, Val %s' % (it.shape, iv.shape)

    train, val = create_iter(it, lt, iv, lv, batch_size=bs)

    output = {
        'train': train,
        'val': val
    }

    if return_raw:
        output['train_img'] = it
        output['train_label'] = lt
        output['val_img'] = iv
        output['val_label'] = lv

    return output
