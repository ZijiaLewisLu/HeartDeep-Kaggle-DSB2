import ipt
import mxnet as mx
import load_data as load
import numpy as np

img, ll, vimg, vll = load.load_pk('/home/zijia/HeartDeepLearning/Net/o1.pk')

def sandwish():
	"""[(1L, 1L, 256L, 256L), (1L, 1L, 3L, 3L), (1L,), (1L, 1L, 254L, 254L)]
 					[(1L, 1L, 254L, 254L)]
	"""
	data = mx.sym.Variable(name = 'data')
	conv = mx.sym.Convolution(data = data, num_filter = 1, kernel = (3,3))
	logis = mx.sym.LogisticRegressionOutput(data = conv, name = 'softmax')
	return logis

def call_back(a):
	print a[0]

def create_model():
	img = np.random.randn(3,1,256,256)
	ll  = np.random.randn(3,1,254,254)

	itr = mx.io.NDArrayIter(
            img,
            label=ll)


	model = mx.model.FeedForward.create(
			sandwish(),
			itr,
			num_epoch = 10,
			batch_end_callback = call_back
		)

if __name__ == '__main__':
	# net = sandwish()

	create_model()