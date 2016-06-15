import ipt, time, os
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
import matplotlib.pyplot as plt

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
        self.name = name

    def __call__(self, epoch, symbol, arg_params, aux_params, acc):
        self.acc_hist[epoch] = acc
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

    def reset(self):
        self.acc_hist = {}
