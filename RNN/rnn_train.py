import ipt
import mxnet as mx
from rnn import rnn
from HeartDeepLearning.solver import Solver
import my_utils as mu
from rnn_load import get

PARAMS={
    'ctx':mu.gpu(2),
    'learning_rate':5,
    'num_epoch':15,
    'initializer':mx.initializer.Xavier(rnd_type='gaussian'),
}

SOLVE = {
    'save_best':True,
    'is_rnn'   :True,  
}


def train(param = PARAMS, sv=SOLVE, small=False):

    sv['name'] = 'TEST'
    input_var = raw_input('Are you testing now? ')
    
    if 'no' in input_var:
        sv.pop('name')
    else:
        sv['name'] += input_var


    net = rnn()
    out = get(2, rate=0.2) 
    train, param['eval_data'] = out['train'], out['val']  
    param['marks'] = param['e_marks'] = out['marks'] 

    s = Solver(net, train, sv, **param)
    s.train()
    s.predict()
    s.all_to_png()
    s.save_best_model()
    s.plot_process()

if __name__ == '__main__':
    # temperal setting
    SOLVE['load'] = False
    SOLVE['load_perfix'] = '/home/zijia/HeartDeepLearning/RNN/Result/<4-22:15:11>TEST[E10]/[ACC-0.03425 E9]'
    SOLVE['load_epoch'] = 9
    #SOLVE['use_logis'] = True
    #SOLVE['block_bn'] = True
    
    PARAMS['num_epoch'] = 10
    PARAMS['learning_rate'] = 15
    # PARAMS['optimizer'] = 'adam'
    # PARAMS['learning_rate'] = 1e-2

    train()