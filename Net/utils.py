import ipt, time, os
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
import matplotlib.pyplot as plt
import os, pickle as pk


class Sfmx(mx.operator.CustomOp):

    # def __init__(self):
        # super(IOU,self).__init__(self)
        # self.First = True

    def forward(self, is_train, req, in_data, out_data, aux):

        #Do nothing!

        self.assign(out_data[0],req[0],in_data[0])
        # print 'out forward'
        # assert False, 'here'

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        
        is_label  = in_data[1]

        not_label = (is_label == 0)
        # out  = out_data[0]

        pred = in_data[0]

        one = mx.ndarray.ones(1, ctx = mx.context.gpu(1))
        e   = mx.ndarray.exp(one)

        base  = mx.ndarray.exp(pred) + mx.ndarray.exp(1-pred)\

        # print 'before grad_is'

        grad_is = e*2/(base*base)

        # exp  = nd.exp(2*pred-1)

        grad_not = - grad_is

        out = - is_label*grad_is - not_label*grad_not

        self.assign(in_grad[0], req[0], out)


@mx.operator.register("sfmx")
class SfmxProp(mx.operator.CustomOpProp):
    # def __init__():
        # super(IOUProp,self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data','label']

    def list_outputs(self):
        return ['outputs']

    def infer_shape(self, inshape):
        ''' [data shape, label shape] ,[output shape], [aux ..?]'''
        # print inshape, 'iou'
        data_shape = inshape[0]
        # print data_shape

        # print [tuple(inshape[0]), tuple(inshape[0])], [ (inshape[0][0], )], []
        return [inshape[0],inshape[0]], [ inshape[0] ] , []

    def create_operator(self, ctx, shapes, dtypes):
        return Sfmx()


def eval_iou(label, pred):
    '''don't know why, but input as np arrays'''
    # assert isinstance(pred, mx.ndarray.NDArray), type(label)
    conjunct = pred * label
    union    = pred + label

    out      = np.sum(conjunct*2)/np.sum(union)

    # print pred
    # print label
    # print out.dtype

    assert 0<=out<=1, 'eval error >> %f' % (out)

    return out

class Callback():

    def __init__(self, name = None):
        self.acc_hist = {}
        self.arg  = {}
        self.name = name

    def __call__(self, epoch, symbol, arg_params, aux_params, acc):
        self.acc_hist[epoch] = acc
        print epoch, self.acc_hist.keys()
        self.arg[epoch] = arg_params
        print 'Epoch[%d] Train accuracy: %f' % ( epoch, np.sum(acc)/float(len(acc)) )
        # print acc
        # print symbol 
        # print arg_params.keys() 
        # print aux_params, '\n\n\n\n\n'

    def get_dict(self):
        return  self.acc_hist
    
    def get_list(self):
        l = []
        for k in sorted(self.acc_hist.keys()):
            print k
            l += self.acc_hist[k]
        return l

    def each_to_png(self):
        if self.name  == None:
            now = time.ctime(int(time.time()))
            now = now.split(' ')
            prefix = now[2]+'-'+now[3]
        else:
            prefix = self.name

        prefix = os.path.join('Img', prefix)
        os.mkdir(prefix)

        for k in sorted(self.acc_hist.keys()):
            plt.plot(self.acc_hist[k])
            path = os.path.join(prefix, str(k)+'.png')
            plt.savefig( path )
            plt.close()

    def reset(self):
        self.acc_hist = {}


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

    train, val = create_iter(*data, batch_size = bs)

    output = {
        'train': train,
        'val'  : val
        }

    if return_raw:
        output['train_img'] = data[0]
        output['train_label'] = data[1]
        output['val_img'] = data[2]
        output['val_label'] = data[3]

    return output

def get(bs, small = False, return_raw = False):

    if small:
        f = "/home/zijia/HeartDeepLearning/Net/o1.pk"
    else:
        f = 'home/zijia/HeartDeepLearning/Net/online.pk'

    return load(f,bs = bs, return_raw = return_raw)   


class IOU(mx.operator.CustomOp):

    # def __init__(self):
        # super(IOU,self).__init__(self)
        # self.First = True

    def forward(self, is_train, req, in_data, out_data, aux):
        # do nothing
        
        # if self.First:
        # print 'in forward'
            # self.First = False
        self.assign(out_data[0],req[0],in_data[0])
        # print 'out forward'
        # assert False, 'here'

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        
        pred = in_data[0]
        ll   = in_data[1]

        out = nd.add(pred, ll)
        out = nd.divide(ll, out)
        # out = (ll/(pred+ll))**2
        out = - nd.multiply(out,out)
        self.assign(in_grad[0],req[0],out)
        # print 'out backward'
        
@mx.operator.register("iou")
class IOUProp(mx.operator.CustomOpProp):
    # def __init__():
        # super(IOUProp,self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data','label']

    def list_outputs(self):
        return ['outputs']

    def infer_shape(self, inshape):
        ''' [data shape, label shape] ,[output shape], [aux ..?]'''
        # print inshape, 'iou'
        data_shape = inshape[0]
        # print data_shape

        # print [tuple(inshape[0]), tuple(inshape[0])], [ (inshape[0][0], )], []
        return [inshape[0],inshape[0]], [ inshape[0] ] , []

    def create_operator(self, ctx, shapes, dtypes):
        return IOU()