import ipt
import mxnet as mx
import basic_right_shape as b
import load_data as load
import create_train_modle as ctm

def lnet():
	return mx.sym.LogisticRegressionOutput(b.out, name = 'softmax')

def callback(l):
	print 'callback' , l[0]

def train_l():

	train, val = load.get_()

	net = lnet()

	model = mx.model.FeedForward(
		net,
        num_epoch = 100
		)

	print 'here'

	model.fit(
		train,
		eval_data = val,
        eval_metric = 'acc', 
        # num_epoch = 1000,
        batch_end_callback = callback,
		)

def no_iter():
	img, ll, vimg, vll = load.load_pk('/home/zijia/HeartDeepLearning/Net/o1.pk')

	model = mx.model.FeedForward(
		lnet(),
		numpy_batch_size = 1,
		)

	model.fit(img,ll,
		batch_end_callback = callback,
		num_epoch = 100
		)

def main():
	# no_iter()
	train_l()

if __name__ == '__main__':
	main()