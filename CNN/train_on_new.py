import ipt, logging
import mxnet as mx
from cnn import cnn_net
import my_utils as u
from solver import Solver
import os
from HeartDeepLearning.RNN.rnn_load import load_rnn_pk, files

PARAMS={
    'ctx':u.gpu(2),
    'learning_rate':3,
    'num_epoch':15,
    #'optimizer':'adam',
    'initializer':mx.initializer.Xavier(rnd_type='gaussian'),
    'wd':1,
}

SOLVE = {
    'save_best':True,
    'is_rnn'   :False,  
}

def train(param=PARAMS, sv=SOLVE, small=False):

    sv['name'] = 'TEST'
    input_var = raw_input('Are you testing now? ')
    
    if 'no' in input_var:
        sv.pop('name')
    else:
        sv['name'] += input_var


    #out = u.get(6,small=True, aug=True) 
    imgs, ll = load_rnn_pk(files)
    imgs = imgs.reshape((-1,1,256,256))
    ll   = ll.reshape((-1,1,256,256))
    datas = u.prepare_set(imgs, ll)

    out = u.create_iter(*datas, batch_size=5)
    net = cnn_net(
        use_logis=True
        )

    param['eval_data'] = out[1] 
  
    s = Solver(net, out[0], sv, **param)
    s.train()
    s.predict()
    s.all_to_png()
    s.save_best_model()
    s.plot_process()

if __name__ == '__main__':
    # temperal setting
    #SOLVE['load'] = True
    #SOLVE['load_perfix'] = '/home/zijia/HeartDeepLearning/Net/CNN/Result/<1-15:28:48>[E40]/[ACC-0.92596 E38]'
    #SOLVE['load_epoch'] = 38
    #SOLVE['use_logis'] = True
    #SOLVE['block_bn'] = True
    
    PARAMS['num_epoch'] = 10
    # PARAMS['optimizer'] = 'adam'
    # PARAMS['learning_rate'] = 1e-2

    train()