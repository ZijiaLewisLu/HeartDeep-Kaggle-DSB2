import ipt
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd

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

    assert 0<out<1, 'eval error'

    return out