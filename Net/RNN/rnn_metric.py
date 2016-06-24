import ipt
from mxnet.metric import EvalMetric
import mxnet as mx

class RnnM(EvalMetric):
    """Custom evaluation metric that takes a NDArray function.

    Parameters
    ----------
    feval : callable(label, pred)
        Customized evaluation function.
    name : str, optional
        The name of the metric
    allow_extra_outputs : bool
        If true, the prediction outputs can have extra outputs.
        This is useful in RNN, where the states are also produced
        in outputs for forwarding.
    """
    def __init__(self, feval, name=None, allow_extra_outputs=True):
        if name is None:
            name = feval.__name__
            if name.find('<') != -1:
                name = 'custom(%s)' % name
        super(RnnM, self).__init__(name)
        self._feval = feval
        self._allow_extra_outputs = allow_extra_outputs

    def update(self, labels, preds):
        # if not self._allow_extra_outputs:
        #     check_label_shapes(labels, preds)

        # print 'type of label and pred', type(preds), type(labels)

        pred = preds[0] if isinstance(preds,list) else preds
        label = labels[0] if isinstance(labels,list) else labels
        # print pred.shape
        # print label.shape
        label = label.asnumpy()
        pred = pred.asnumpy()

        reval = self._feval(label, pred)
        # print reval
        # print self._feval(label[0,0], pred[0,0])

        # print reval
        if isinstance(reval, tuple):
            (sum_metric, num_inst) = reval
            self.sum_metric += sum_metric
            self.num_inst += num_inst
        else:
            self.sum_metric += reval
            self.num_inst += 1