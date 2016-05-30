from PIL import Image as Img
import numpy as np

w, h = 512, 512
data = np.zeros((h, w), dtype=np.uint8)
data[256, 256] = 255
img = Img.fromarray(data, 'L')
img.save('my.png')
