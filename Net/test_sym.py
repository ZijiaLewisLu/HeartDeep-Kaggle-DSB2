import ipt
import mxnet as mx
import numpy as np
import os
os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"

class IOU(mx.operator.CustomOp):

    # def __init__():
    #     super(IOU,self).__init__(self)

    def forward(self, is_train, req, in_data, out_data, aux):
        # do nothing
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

def make_iou(data, label):
    return mx.sym.Custom(data =data, label = label, name = 'iou', op_type='iou')

def get_iou():
    d = mx.sym.Variable(name = 'data')
    # l = mx.sym.Variable(name = 'label')

    return mx.sym.Custom(data = d, name = 'iou', op_type='iou')

def callback(l):
    print '\n>>>>>callback', l[0]
    print l[1:], '\n'


if __name__ == '__main__':

    d = mx.sym.Variable(name = 'data')
    l = mx.sym.Variable(name = 'label')

    iou = mx.sym.Custom(data =d, name = 'softmax', op_type='iou')

    img = np.random.randn(10,1,256,256)
    label = np.random.rand(10,1,256,256)

    vimg = np.random.randn(2,1,256,256)
    vlabel = np.random.randn(2,1,256,256)

    model = mx.model.FeedForward(iou, num_epoch=10)

    itr   = mx.io.NDArrayIter(img, label = label, batch_size = 1)
    viter = mx.io.NDArrayIter(vimg, label = vlabel, batch_size = 1)

    print 'start to train'
    model.fit(
        itr,
        eval_data = viter,
        eval_metric = 'acc',
        batch_end_callback = callback
        )
    
