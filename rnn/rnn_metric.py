import ipt
import mxnet as mx
from mxnet.metric import EvalMetric
import numpy as np
import my_utils as mu

class rnnM(EvalMetric):
    
    def update(self, labels, preds):
   
        labels = labels[0].asnumpy()
        T = labels.shape[1]
        assert T==len(preds)
        labels = np.split(labels, T, axis=1) 
        
        acc_all = []
        for p , l in zip(preds, labels):
            p = p.asnumpy()
            l = l[0]
            print l.shape, p.shape
            acc = mu.eval_iou(l, p)
            acc_all.append(acc)

        average = sum(acc_all)/float(len(acc_all))
        self.sum_metric += average
        self.num_inst += 1

