import ipt
import mxnet as mx
from mxnet.metric import EvalMetric

import my_utils as mu

class rnnM(EvalMetric):
    
    def update(self, labels, preds):
    
        if not isinstance(labels, list):
            labels = [labels]
        if not isinstance(preds, list):
            preds =  [preds]

        acc_all = []
        for l, p in zip(labels, preds):
            l = l.asnumpy()
            p = p.asnumpy()

            acc = mu.eval_iou(l, p)
            acc_all.append(acc)
    
        average = float(sum(acc_all))/len(acc_all)
        
        self.sum_metric += average
        self.num_inst += 1

