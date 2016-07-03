import numpy as np
import os, pickle as pk


def _load_pk_file(fname, rate, shift = False):
    with open(fname,'r') as f:
        img = pk.load(f)
        ll  = pk.load(f)

    img = img.astype(np.float64)
    img -= img.mean(axis = 0)

    if shift == True:
        # left-right
        N = img.shape[0]

        # img = np.roll(img, 30, axis=2)
        # ll = np.roll(ll, 30, axis=2)

        for i in range(N):
            img[i, : , :] = np.rot90(img[i, : , :])
            ll[i, : , :] = np.rot90(ll[i, : , :])

        img = np.roll(img, 50, axis=2)
        ll = np.roll(ll, 50, axis=2)

    N = img.shape[0]

    img = img[:, None, :, :]
    ll = ll[:, None, :, :]

    p = int(rate * N)

    val_img = img[:p]
    img     = img[p:]
    val_ll  = ll[:p]
    ll      = ll[p:]

    return img, ll, val_img, val_ll

def load_pk(fname, rate = 0.1, shift = False):
    
    fname = [fname] if not isinstance(fname, list ) else fname

    img_train = ll_train = img_val = ll_val = None 
    for f in fname:
        if shift == True:
            img, ll , vimg, vll = _load_pk_file(f, rate, shift = True)
        else:
            img, ll , vimg, vll = _load_pk_file(f, rate)

        if img_train == None:
            img_train = img
        else:
            img_train = np.concatenate((img_train,img))

        if ll_train == None:
            ll_train = ll
        else:
            ll_train = np.concatenate((ll_train,ll))

        if img_val == None:
            img_val = vimg
        else:
            img_val = np.concatenate((img_val,vimg))

        if ll_val == None:
            ll_val = vll
        else:
            ll_val = np.concatenate((ll_val,vll))

    # return img_train , ll_train, img_val, ll_val
    # print 'len of val', img_val.shape[0]

    return img_train, ll_train, img_val, ll_val

def load(filename, bs = 10, return_raw = False, shift = False):
    
    data = load_pk(filename, shift = shift)

    if return_raw:
        output = {
        'train_img': data[0],
        'train_label' : data[1],
        'val_img' : data[2],
        'val_label' : data[3]
        }

    return output

def get(bs, small = False, return_raw = True, shift = False, new = False):
    if small:
        f = "/home/zijia/HeartDeepLearning/Net/data/o1.pk"

        # f = '/home/weilin/Desktop/New_DATA/'
    else:
        if new == True:
            fname = '/home/weilin/Desktop/New_DATA/'
            f = []
                # # folder ver
            ls = os.listdir(fname)
            for i in ls:
                if '.pk' in i:
                    f.append(os.path.join(fname,i))
        else:
            f = [
                    '/home/zijia/HeartDeepLearning/Net/data/online.pk',
                    '/home/zijia/HeartDeepLearning/Net/data/validate.pk',
                ]
                
    data = load_pk(f, shift = shift)   
    
    from my_utils import create_iter
    train, val = create_iter(*data, batch_size=bs)
    return {'train':train, 'val':val}



