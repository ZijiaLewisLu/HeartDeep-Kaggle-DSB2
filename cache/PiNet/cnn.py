import ipt
from model import model
import minpy as mp
import minpy.numpy as np
from collections import OrderedDict


def twolayers(X,y,params):
	p = params

	plist = params.values()

	def forward(X,y,*p):
		N, C, H, W = X.shape

		X = X.reshape((N,C*H*W))

		print '>>',X.shape
		print '>>',p[0].shape

		first = np.dot( X, p[0] ) + p[1]

		second = np.dot( first, p[2] ) + p[3]

		exp = np.exp(second)

		pred = exp / np.sum(exp)

		N = X.shape[0]

		loss = -np.sum( pred[np.arange(N),y] )

		return loss

	grad_loss = mp.core.grad_and_loss(forward, argnum = (2,3,4,5))

	return grad_loss(X,y,*plist)


def test():
	X = np.random.randn(3,3,5,5)
	y = np.random.randint(3,size = (3,))

	print y.shape

	params = OrderedDict()
	params['one_weight'] = (3*5*5, 10)
	params['one_bias']   = (10,)
	params['two_weight'] = (10,3)
	params['two_bias']   = (3,)

	print 'params', params
	from utils import gaussic

	p = OrderedDict()
	for k in params:
		p[k] = gaussic(params[k],1)
		# print p[k].asnumpy().shape

	print 'p', p

	print twolayers(X,y,p)


if __name__ == '__main__':
	test()