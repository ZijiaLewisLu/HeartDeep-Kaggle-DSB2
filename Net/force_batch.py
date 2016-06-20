import ipt
import mxnet as mx
import mxnet.ndarray as nd

'''UNFINISHED'''

class Batch(mx.operator.CustomOp):

    # def __init__(self):
        # super(IOU,self).__init__(self)
        # self.First = True

    def forward(self, is_train, req, in_data, out_data, aux):

        #Do nothing!
        fea = in_data[0]

        N,C,H,W = fea.shape

        mean = nd.sum(fea) / (N*C*H*W)
        std =  nd.squre(nd.fea-mean)/ (N*C*H*W)

        fea = fea/std - mean

        self.assign(out_data[0], req[0], fea)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		
		        



@mx.operator.register("batch")
class BatchProp(mx.operator.CustomOpProp):
    # def __init__():
        # super(IOUProp,self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['outputs']

    def infer_shape(self, inshape):
        ''' [data shape, label shape] ,[output shape], [aux ..?]'''
        # print inshape, 'iou'
        # print data_shape

        # print [tuple(inshape[0]), tuple(inshape[0])], [ (inshape[0][0], )], []
        return [ inshape[0] ], [ inshape[0] ] , []

    def create_operator(self, ctx, shapes, dtypes):
        return Batch()