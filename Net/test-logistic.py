import ipt
import mxnet as mx
import basic_right_shape as b
import load_data as load
import create_train_modle as ctm

def lnet():
	return mx.sym.LogisticRegressionOutput(b.out, name = 'logist')

def callback(l):
	print l[0]

def train_l():
	train, val = load.get_()

	net = lnet()

	model = mx.model.FeedForward(net)

	print 'here'

	model.fit(
		train,
		eval_data = val,
        eval_metric = 'acc', 
        # num_epoch = 1000,
        batch_end_callback = callback
		)


def main():
	train_l()

if __name__ == '__main__':
	main()