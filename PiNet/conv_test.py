import ipt
import minpy.numpy as np
import time

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    Hout = 1 + (H + 2 * pad - HH) / stride
    Wout = 1 + (W + 2 * pad - WW) / stride
    # print 'N:%d,C:%d,H:%d,W:%d,F:%d,HH:%d,WW:%d,Hout:%d,Wout:%d,pad:%d,stride:%d' \
    #       % (N, C, H, W, F, HH, WW, Hout, Wout, pad, stride)

    # row_w shape: (F, C * HH * WW)

    print w.shape
    row_w = w.reshape((F, C*HH*WW))
    print row_w.shape
    # pad_x shape: (N, C, H + 2 * pad, W + 2 * pad)
    pad_x = np.pad(x, pad, 'constant', constant_values=0)
    if pad != 0:
        pad_x = pad_x[pad:-pad, pad:-pad]

    print 'pad_x', pad_x.shape
    out = np.zeros((N, F, Hout, Wout))
    # column_x shape: (N, C * HH * WW, Hout * Wout)
    for filter_W in range(Wout):
        for filter_H in range(Hout):
            block = pad_x[:, :,
                    filter_H * stride:filter_H * stride + HH,
                    filter_W * stride:filter_W * stride + WW]
            N, C, H, W = block.shape

            # print block.shape
            block = block.reshape((N, C*H*W))

            # print block.shape
            # print row_w.shape

            o = np.dot(block, row_w.T)
            # print type(o)
            o  = np.copy(o)
            b  = np.copy(b)

            out[:, :, filter_H, filter_W] = o + b
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache

def main():
    X = np.random.randn(10,3,256,256)
    w = np.random.randn(8,3,3,3)
    b = np.zeros((8,))
    params = {
        'pad':1,
        'stride':2
    }

    start = time.time()

    conv_forward_naive(X,w,b,params)

    print time.time() - start

if __name__ == '__main__':
    main()

