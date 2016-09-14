import ipt
import mxnet as mx
from mxnet.metric import EvalMetric
import numpy as np
import my_utils as mu

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import numpy as np

path = os.path.join(os.environ['HOME'], 'insight_pic')
# os.mkdir(os.path.join(os.environ['HOME'], 'insight_pic'))

class rnnM(EvalMetric):
    
    def update(self, labels, preds):
   
        labels = labels[0].asnumpy()
        T = labels.shape[1]
        assert T==len(preds)
        labels = np.split(labels, T, axis=1) 
        
        acc_all = []
        idx = 0
        for p , l in zip(preds, labels):
            idx += 1
            p = p.asnumpy()
            l = l[0]
            acc = mu.eval_iou(l, p)
            acc_all.append(acc)
            
            if self.num_inst % 1 == 0:
                L = p.shape[-1]
                gap = np.zeros((5,L))
                
                pic = np.concatenate([p[0,0], gap, l[0,0]], axis=0)                                
                plt.imshow(pic)
                
                plt.savefig(os.path.join(path, '%d_%d.png'%(self.num_inst, idx)))
                plt.close()
            

        average = sum(acc_all)/float(len(acc_all))
        self.sum_metric += average
        self.num_inst += 1
        