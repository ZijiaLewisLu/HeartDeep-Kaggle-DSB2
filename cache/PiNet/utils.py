import ipt
import minpy
import minpy.numpy as np

def gaussic(shape, weight_scale):
	return np.random.randn(*shape)*weight_scale