from maker import Heart, Maker
from maker import RE
import numpy as np
from copy import copy

space = np.random.randn(4)*0.18+1
print space
img_all = []
label_all= []
for s in space:
    require = copy(RE)
    require[0] = [ int(round(n*s)) for n in require[0] ]
    require[1] = [ int(round(n*s)) for n in require[1] ]
    center_scale = np.random.choice(space, 1)
    center = (int(round(128*center_scale)), int(round(128*center_scale)))
    print center, require

    m = Maker(require)
    a = np.random.randint(360)
    m.generate(5, a, center=center)
        
    img = [ i[None,None,None,:,:] for i in m.imgs ]
    label = [ l[None,None,None,:,:] for l in m.labels]
    img = np.concatenate(img, axis=0)
    label = np.concatenate(label, axis=0)
    img_all.append(img)
    label_all.append(label)

img_all = np.concatenate(img_all, axis=1)
img_all =   np.transpose(img_all, (1,0,2,3,4))
label_all = np.concatenate(label_all,axis=1)
label_all = np.transpose(label_all, (1,0,2,3,4))

print img_all.shape
assert img_all.shape[-2:] == (256,256)

import pickle as pk
from HeartDeepLearning.my_utils import parse_time

fname = '[N%d,T%d]'%(img_all.shape[0], img_all.shape[1]) +parse_time()+'.pk'
with open(fname,'w') as f:
    pk.dump(img_all,f)
    pk.dump(label_all,f) 

