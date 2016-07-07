import ipt
import mxnet as mx
import matplotlib.pyplot as plt
from cnn_internal import fetch_internal
from cnn import cnn_net
import my_utils as mu
import os

p = '/home/zijia/HeartDeepLearning/CNN/Result/<0Save>/<1-17:12:45>[E40]/[ACC-0.92900 E39]'
e = 39
net = cnn_net()
val = mu.get(2,small=True)['val']

outputs, imgs, lls = fetch_internal(net,val,p,e)


stamp = 'Inspect/'+mu.parse_time()+'/'
os.makedirs(stamp)

mu.save_img(imgs[0,0],stamp+'Input')
#for k,v in outputs.items():
#    if len(v.shape)>2:
#        mu.save_img(v[0,0], stamp+k)

