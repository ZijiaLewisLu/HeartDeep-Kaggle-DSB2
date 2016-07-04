from main import Heart, Maker
from main import RE
import numpy as np

space = np.linspace(0.5,1.2,10)
for s in space:
    require = RE.copy()
    require[0] = require[0]*s
    center_scale = np.random.choice(space, 1)
    center = (256*center_scale, 256*center_scale)

    m = Maker(require)
    imgstack = None
    labelstack=None
    for a in np.random.randn(360, size=10):
        m.generate(30,a)
        if stack is None:
            imgstack=


    m.generate(30,)