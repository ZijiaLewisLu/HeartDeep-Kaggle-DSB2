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
    out = get(2, rate=0.2, small=True) 
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
    SOLVE['load'] = True
    SOLVE['load_from_cnn'] = False

    SunnyCNN=('/home/zijia/HeartDeepLearning/CNN/Result/<0Save>/<1-17:12:45>[E40]/[ACC-0.92900 E39]', 39)
    NewCNN  =('/home/zijia/HeartDeepLearning/CNN/Result/<0Save>/<6-11:38:53>NewGood[E30]/[ACC-0.93164 E29]',29)
    NewFromCnn=('/home/zijia/HeartDeepLearning/RNN/Result/<6-12:26:43>TEST[E10]/[ACC-0.17967 E9]',9)

    SOLVE['load_perfix'], SOLVE['load_epoch'] = NewFromCnn
    #SOLVE['use_logis'] = True
    #SOLVE['block_bn'] = True
    
    PARAMS['num_epoch'] = 3
    PARAMS['learning_rate'] = 10
    # PARAMS['optimizer'] = 'adam'
    # PARAMS['learning_rate'] = 1e-2

    train(small=True)