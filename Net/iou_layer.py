import ipt
import mxnet as mx
import numpy as np

class IOU(mx.operator.CustomOp):

    # def __init__():
    #     super(IOU,self).__init__(self)

    def forward(self, is_train, req, in_data, out_data, aux):
        # do nothing
        print 'in'
        self.assign(out_data[0],req[0],in_data[0])
        print 'one forward end'
        # print 'end forward'
        # assert False, 'here'

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        
        pred = in_data[0]
        ll   = in_data[1]

        out = (ll/(pred+ll))**2
        self.assign(in_grad[0],req[0],out)
        

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