import ipt
import mxnet as mx
import net_ as n
import load
import utils as u

d1 = mx.sym.Variable(name = 'data1')
d2 = mx.sym.Variable(name = 'data2')

net1 = n.net(d1)
net2 = n.net(d2)



