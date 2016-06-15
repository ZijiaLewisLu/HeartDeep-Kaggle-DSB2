import ipt
import mxnet as mx
import load_data as load
from basic_right_shape import net

train, val = load.get_()

def test(lr):
	mx.model.Feedforward.create(
		net,
		train,
		eval_data = val,
		eval_metric = mx.metric.create(eval_iou)

		)
