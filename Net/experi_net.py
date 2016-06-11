import ipt
import minpy as minpy
import mxnet as mx
import minpy.numpy as np
import create_train_modle as old
import load_data as load

params = old.Params

def net_basic(pm):
    ''' pm should be a dict of the params of each layers '''
    data = mx.sym.Variable(name= 'data')  #name must be data, don't know why


    conv1 = mx.sym.Convolution(name = 'conv1', data = data, kernel = pm['c1']['fsize'], 
            num_filter = pm['c1']['fnum'], stride = pm['c1']['stride'], pad = pm['c1']['pad'] )
    relu1 = mx.sym.Activation(data = conv1, act_type = 'relu')
    conv2 = mx.sym.Convolution(name = 'conv2', data = relu1, kernel = pm['c2']['fsize'], 
        num_filter = pm['c2']['fnum'], stride = pm['c2']['stride'], pad = pm['c2']['pad'] )
    relu2 = mx.sym.Activation(data = conv2, act_type = 'relu')

    pool1 = mx.sym.Pooling(data = relu2, pool_type = "max", kernel=(2,2), stride = (2,2))


    conv3 = mx.sym.Convolution(name = 'conv3', data = pool1, kernel = pm['c3']['fsize'], 
            num_filter = pm['c3']['fnum'], stride = pm['c3']['stride'], pad = pm['c3']['pad'] )
    relu3 = mx.sym.Activation(data = conv3, act_type = 'relu')
    pool2 = mx.sym.Pooling(data = relu3, pool_type = "max", kernel=(2,2), stride = (2,2))
    

    conv4 = mx.sym.Convolution(name = 'conv4', data = pool2, kernel = pm['c4']['fsize'], 
            num_filter = pm['c4']['fnum'], stride = pm['c4']['stride'], pad = pm['c4']['pad'] )
    relu4 = mx.sym.Activation(data = conv4, act_type = 'relu')
    pool3 = mx.sym.Pooling(data = relu4, pool_type = "max", kernel=(2,2), stride = (2,2))


    conv5 = mx.sym.Convolution(name = 'conv5', data = pool3, kernel = pm['c5']['fsize'], 
            num_filter = pm['c5']['fnum'], stride = pm['c5']['stride'], pad = pm['c5']['pad'] )
    relu5 = mx.sym.Activation(data = conv5, act_type = 'relu')
    conv6 = mx.sym.Convolution(name = 'conv6', data = relu5, kernel = pm['c6']['fsize'], 
        num_filter = pm['c6']['fnum'], stride = pm['c6']['stride'], pad = pm['c6']['pad'] )
    relu6 = mx.sym.Activation(data = conv6, act_type = 'relu')


    up1  = mx.sym.UpSampling(relu6, scale = 2, sample_type= 'bilinear', num_args = 1)

    
    conv7 = mx.sym.Convolution(name = 'conv7', data = up1, kernel = pm['c7']['fsize'], 
        num_filter = pm['c7']['fnum'], stride = pm['c7']['stride'], pad = pm['c7']['pad'] )
    relu7 = mx.sym.Activation(data = conv7, act_type = 'relu')

    up2  = mx.sym.UpSampling(relu7, scale = 2, sample_type = 'bilinear', num_args = 1)
    
    conv8 = mx.sym.Convolution(name = 'conv8', data = up2, kernel = pm['c8']['fsize'], 
        num_filter = pm['c8']['fnum'], stride = pm['c8']['stride'], pad = pm['c8']['pad'] )
    relu8 = mx.sym.Activation(data = conv8, act_type = 'relu')

    up3  = mx.sym.UpSampling(relu3, scale = 2, sample_type = 'bilinear', num_args = 1)


    conv9 = mx.sym.Convolution(name = 'conv9', data = up3, kernel = pm['c9']['fsize'], 
            num_filter = pm['c9']['fnum'], stride = pm['c9']['stride'], pad = pm['c9']['pad'] )
    relu9 = mx.sym.Activation(data = conv9, act_type = 'relu')
    # conv10 = mx.sym.Convolution(name = 'conv10', data = relu9, kernel = pm['c10']['fsize'], 
    #     num_filter = pm['c10']['fnum'], stride = pm['c10']['stride'], pad = pm['c10']['pad'] )
    # relu10 = mx.sym.Activation(data = conv10, act_type = 'relu')


    # conv11 = mx.sym.Convolution(name = 'conv11', data = relu10, kernel = pm['c11']['fsize'], 
    #         num_filter = pm['c11']['fnum'], stride = pm['c11']['stride'], pad = pm['c11']['pad'] )
#    softmax = mx.sym.Softmax(name = 'softmax', data = conv11)
    return relu9

def pred_out():
    rl = net_basic(params)

    conv = mx.sym.Convolution(name = 'conv10', data = rl, kernel = (7,7), num_filter = 1,  
            stride = (1,1), pad = (0,0) )
    return out  = mx.sym.Activation(data = conv, act_type = 'sigmoid')


class IOU(mx.operator.CustomOp):

    def __init__():
        super.(IOU,self).__init__(self)

    def forward(self, is_train, req, in_data, out_data, aux):
        pred = in_data[0]
        ll   = in_data[1]
        # ll = mx.sym.Variable(name = 'label')
        out = 2* mx.sym.sum(pred*ll, axis = 0)/mx.sym.sum(pred + ll, axis = 0).
        self.assign(out_data[0],req[0],out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass
        

@mx.operator.register("iou")
class IOUProp(mx.operator.CustomOpProp):
    def __init__():
        super(IOUProp,self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data','label']

    def list_outputs(self):
        return ['outputs']

    def infer_shape(self, inshape):
        ''' [data shape, label shape] ,[output shape], [aux ..?]'''
        return [inshape[0],inshape[0]], [inshape[0][0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return IOU()


if __name__ == "__main__":

    img, ll, vimg, vll = load.load_pk('/home/zijia/HeartDeepLearning/Net/o1.pk')

    # upper = net_basic(params)
    net = mx.sym.SoftmaxOutput(data = upper, act_type = 'sigmoid', name = 'softmax_label')

    model = mx.model.FeedForward(
                symbol = net,
                ctx = mx.context.gpu(0),
                num_epoch = 1000,
                learning_rate = 3e-3,
                optimizer = 'adam'
                # initializer = init
                )



