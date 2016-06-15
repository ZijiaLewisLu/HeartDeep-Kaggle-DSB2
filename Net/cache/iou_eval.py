def forward(self, is_train, req, in_data, out_data, aux):
    pred = in_data[0]
    ll   = in_data[1]
    # print '\nindata!>', in_data
    # ll = mx.sym.Variable(name = 'label')
    # print 'in forward'
    muti = pred * ll
    union = pred + ll

    # print 'u shape', union.shape

    # print '\n\n\n start upper'
    upper = 2* mx.ndarray.sum(muti, axis = (1,2,3))
    # print 'start lower'
    lower = mx.ndarray.sum(union, axis = (1,2,3))
    # print 'lower shape', lower.shape

    out = upper/lower

    # print 'outshape',out.shape
    # print 'outdata>',out_data[0].shape

    # print 'start assign'
    self.assign(out_data[0],req[0],out)
    print 'one forward end'
    # print 'end forward'
    # assert False, 'here'