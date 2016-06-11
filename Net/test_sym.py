import ipt
import mxnet as mx

class IOU(mx.operator.CustomOp):

    def __init__():
        super(IOU,self).__init__(self)

    def forward(self, is_train, req, in_data, out_data, aux):
        pred = in_data[0]
        ll   = in_data[1]
        # ll = mx.sym.Variable(name = 'label')
        out = 2* mx.sym.sum(pred*ll, axis = 0)/mx.sym.sum(pred + ll, axis = 0)
        self.assign(out_data[0],req[0],out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass
        

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
        return [inshape[0],inshape[0]], [inshape[0][0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return IOU()


if __name__ == '__main__':

    d = mx.sym.Variable(name = 'data')
    l = mx.sym.Variable(name = 'label')

    iou = mx.sym.Custom(data =d, label = l, name = 'iou', op_type='iou')

    
