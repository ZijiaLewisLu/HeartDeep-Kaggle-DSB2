import ipt
import mxnet as mx
import load_data as load

img, ll, vimg, vll = load.load_pk('/home/zijia/HeartDeepLearning/Net/o1.pk')

def sandwish():
	data = mx.sym.Variable(name = 'data')
	conv = mx.sym.Convolution(data = data, num_filter = 1, kernel = (3,3))
	logis = mx.sym.LogisticRegressionOutput(data = conv, name = 'logis')
	return logis

def create_model():
	return None

if __name__ == '__main__':
	net = sandwish()