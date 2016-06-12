import ipt 
import mxnet as mx
import numpy as np

import experi_net as e


model = e.return_model()

data  =  np.ones((10,1,256,256))
ll    =  np.ones((10,1,256,256))

train = mx.io.NDArrayIter(data, label = ll, batch_size = 1)

data_shapes = [('data', (1, 1, 256, 256))]


#to init params
d = train.provide_data
l = train.provide_label

model._init_params(dict(d+l))

pred = model.predict(
	train,
	return_data = True
	)
