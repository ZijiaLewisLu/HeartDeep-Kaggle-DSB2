import ipt
import mxnet as mx
import numpy as np
import mxnet.ndarray as nd

class Sfmx(mx.operator.CustomOp):

    # def __init__(self):
        # super(IOU,self).__init__(self)
        # self.First = True

    def forward(self, is_train, req, in_data, out_data, aux):
        # pred = in_data[0]

        # pred = 1 - 2 * pred

        # out = 1/ ( 1 + pred )

        # nd.concatenate(is_pred, not_pred)

        # is_pred  = nd.exp(is_pred)
        # not_pred = nd.exo(not_pred)

        self.assign(out_data[0],req[0],in_data[0])
        # print 'out forward'
        # assert False, 'here'

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        
        is_label  = in_data[1]

        not_label = (is_label == 0)
        # out  = out_data[0]

        pred = in_data[0]

        one = mx.ndarray.ones(1)
        e   = mx.ndarray.exp(one)

        base  = mx.ndarray.exp(pred) + mx.ndarray.exp(1-pred)

        grad_is = e*2/(base*base)

        # exp  = nd.exp(2*pred-1)

        grad_not = - grad_is

        out = is_label*grad_is + not_label*grad_not

        self.assign(in_grad[0], req[0], out)

@mx.operator.register("sfmx")
class Sfmx(mx.operator.CustomOpProp):
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
