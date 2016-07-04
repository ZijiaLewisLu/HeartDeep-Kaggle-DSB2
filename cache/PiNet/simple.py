import ipt
import minpy as mp
import minpy.numpy as np

from fc_net_minpy import TwoLayerNet
from solver import Solver

def main():

	X = np.random.randn(15,3,32,32)
	y = np.random.randint(10,size = (15,))

	data = {
	'X_train': X,
	'y_train': y,
	'X_val':   X,
	'y_val': y,
	}

	s = Solver(
		TwoLayerNet(),
		data,
		num_epochs = 5,
		print_every=1,
		# verbose = False
		)
	s.train()


if __name__ == '__main__':
	main()