# coding: utf-8
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import skimage.draw as d
import skimage.transform as t
import time
RE = [
    [50,10], # outter radius and width of circle
    [52,33]  # h and w of ellipse
    ]
INNER = 130
OUTER = 80

def show(img):
    plt.imshow(img, cmap = 'gray')
    plt.show()
    plt.close()

class Heart():
    def __init__(self, RE):
        self.base_lv_radius, self.base_lv_thick = RE[0]
        self.base_e_h, self.base_e_w = RE[1]

        v = max(self.base_e_h, self.base_lv_radius)
        self.v = 2*v +20
        self.h = 2*self.base_lv_radius + self.base_e_w + 20
        self.canva = np.zeros((self.v,self.h))

    def find_nbor(self, point, ctr):
        target = []
        x, y = ctr
        px, py = point[0], point[1]
        for i, pair in enumerate(zip(x,y)):
            _x, _y = pair 
            px, py = point[0], point[1]
            array  = np.array([_x, _y])
            # print point, array
            if _x<=px+1 and _x>=px-1 and _y<=py+1 and _y>=py-1 and (py!=_y or px!=_x):
                already = False
                for i in target:
                    if (array==i).all(): already = True
                
                if not already: target.append(array)
                              
        assert len(target)==2, "WTF? target{0} point{1}".format(target, point)
        return target
    
    def random_center(self, radius, ctr, point = None):
        """find the center of a circle tangent to outer contour"""
        N = len(ctr[0])
        good = False
        
        if point is None:
            start = np.random.choice(N-1)
            point  = np.array([ctr[0][start], ctr[1][start]])

        dots = zip(ctr[0], ctr[1])
        while not good:
            # random shift point from origin
            point = point + np.random.randn()*10
            distance = [ np.linalg.norm(x-point) for x in dots]
            idx = np.argmin(distance)
            point = dots[idx]
            
            #locate the center
            vec   = self.find_nbor(point, ctr)
            vec   = vec[0] - vec[1]
            dis   = np.linalg.norm(vec)
            Y = abs(round(radius*vec[1]/dis))
            X = abs(round(radius*vec[0]/dis))
            if midl[0]>self.h:
                Y = midl[0]-Y
            else:
                Y = midl[0]+Y
            if midl[1]>self.w:
                X = midl[1]-X
            else:
                X = midl[1]+X
        
            if X-radius>0 and X+radius<2*self.w and Y-radius>0 and Y+radius<2*self.h:
                good = True   

        center = np.array((Y,X))
        radius_vec = center - point 
        return center, radius_vec
    
    def add_LV(self, scale, drift=10):
        radius = self.lv_radius = self.base_lv_radius * scale
        thick  = self.lv_thick  = self.base_lv_thick * (4+scale)/5
        
        # get center
        shift = np.array([np.random.randn()*drift, radius])
        
        # print shift
        center = np.array([self.v/2, 10]) + shift
        self.lv_center = center
        self.lv_radius_vec = np.array([0,radius])

        # get points
        self.lv_big = d.circle(center[0], center[1], radius)
        self.lv_small = d.circle(center[0], center[1], radius-thick)
        #self.lv_ctr = d.circle_perimeter(int(center[0]), int(center[1]), int(radius))
    
    def add_Ellipse(self, scale = 1, thick = 5):
        """h for vertical， w for horizontal"""
        h = self.e_h = self.base_e_h * scale
        w = self.e_w = self.base_e_w * scale
        
        # e and lv should not be too close
        shift = self.e_w/8
        center = self.lv_center + self.lv_radius_vec
        center[1] += shift
        
        self.e_center = center 
        self.e_thick = thick
        self.e_big  = d.ellipse(center[0], center[1], h, w)
        self.e_small= d.ellipse(center[0], center[1], h-thick, w-thick)
        # self.ctr = d.ellipse_perimeter(h+5, w+5, h, w)
        
    def add_background(self):
        dis = self.lv_radius**2/self.e_h
        poly = np.array((
        (self.e_center[0]+self.e_h, self.e_center[1]),
        (self.e_center[0]-self.e_h, self.e_center[1]),
        (self.lv_center[0]-self.lv_radius, self.lv_center[1]),
        (self.lv_center[0]+self.lv_radius, self.lv_center[1]),
        ))
        self.back = d.polygon(poly[:, 0], poly[:, 1])
        
    
    def draw_LV(self):
        self.canva[self.lv_big[0],   self.lv_big[1]]   = np.random.randn(self.lv_big[0].size)*5+OUTER
        self.canva[self.lv_small[0], self.lv_small[1]] = np.random.randn(self.lv_small[0].size)*10+INNER

    def draw_E(self):
        self.canva[self.e_big[0], self.e_big[1]]   = np.random.randn(self.e_big[0].size)*5+OUTER
        self.canva[self.e_small[0], self.e_small[1]] = np.random.randn(self.e_small[0].size)*10+INNER
        
    def draw_background(self):
        self.canva[self.back[0], self.back[1]] = np.random.randn(self.back[0].size)*5+OUTER
    
    def make_label(self):
        bg = np.zeros_like(self.canva)
        bg[self.lv_small[0], self.lv_small[1]] = 1
        self.label = bg
        
    def show(self):
        plt.imshow(self.canva, cmap = 'gray')
        plt.show()
        #u.figure.clear()
        #plt.clf()
        plt.close()
        
    def clean(self):
        self.canva = np.zeros((self.v,self.h))

    def _make(self, scale):
        self.add_LV(scale, drift=1)
        self.add_Ellipse()
        self.add_background()
        self.make_label()
            
        self.draw_background()
        self.draw_E()
        self.draw_LV()
        #self.show()
    
    def make(self, scales=None):
        self.series = []
        self.labels  = []
        if scales is None:
            scales = [1]
        
        for s in scales:
            self.clean()
            self._make(s)
            self.series.append(self.canva)
            self.labels.append(self.label)

    def move(self, base, num, plot = False):
        base = 1.0/base
        kkk = np.linspace(base*np.pi ,(1-base)*np.pi, num=num)
        scale = np.sin(kkk)
        
        
        if plot:
            plt.plot(scale)
            plt.show()
            
        self.make(scale)
        
        assert len(self.series)==num, "LENTH ERROR {}".format(len(self.series))

        if plot:
            for i in range(num):
                show(self.series[i])
                show(self.labels[i])
                
        return self.series, self.labels



class Maker(object):
    
    def __init__(self, re, small=True):
        """RE has the info for LV and E"""
        self.RE = re
        self.heart = Heart(re)
        self.background = misc.imread('background_small.jpeg', mode='P') if small else misc.imread('background.jpg', mode='P')
        self.small = small
    
    def rotate(self, img, label, angle):
        assert img.shape==label.shape
        
        img =   t.rotate(img, angle, resize=True)
        label = t.rotate(label, angle, resize=True)
        assert img.shape==label.shape

        lower_x, lower_y = 0 , 0
        while (img[lower_x,:]==0).all(): 
            lower_x +=1    
        while (img[:, lower_y]==0).all():
            lower_y +=1

        upper_x, upper_y = img.shape
        upper_x -=1
        upper_y -=1
        while (img[upper_x,:]==0).all():
            upper_x -=1
        while (img[:, upper_y]==0).all():
            upper_y -=1

        img = img[lower_x:upper_x, lower_y:upper_y]
        label = label[lower_x:upper_x, lower_y:upper_y]

        return img, label
    
    def combine(self, heart, label, center):
        if center is None:
            x, y = self.background.shape
            center = (round(x/2),round(y/2))
        
       
        hx, hy = heart.shape
        x, y = center
        x = x + np.random.randn()*4 - hx/2
        y = y + np.random.randn()*4 - hy/2
        x = int(round(x))
        y = int(round(y))
       
        #make image
        mask = (heart!=0)
        img = self.background.copy()
        img[x:x+hx, y:y+hy][mask] = heart[mask]
        
        #make label
        ll = np.zeros_like(img)
        ll[x:x+hx, y:y+hy] = label 
        return img, ll
    
    
    def generate(self, num_pic, angle, base=6, plot=False, center=None, shape=None, downsample = False):
        shape = 300 if shape is None else shape
        if not isinstance(shape, tuple):
            shape = int(round(shape+np.random.randn()*shape/100))
        else:
            shape = tuple([ int(round(x+np.random.randn()*x/100))
                                                  for x in shape ])

            
        series, labels = self.heart.move(base, num_pic)
        self.imgs = []
        self.labels = []
        for i in range(num_pic):
            
            img, label = series[i], labels[i]
            assert img.mean() > 0
            if not isinstance(shape, tuple):
                assert isinstance(shape, int), shape
                s = shape
                m_x = np.max(img.shape)
                idx = np.argmax(img.shape)
                scale = s/m_x
                shape = [0,0]
                shape[idx] = s
                shape[1-idx] = int(round(img.shape[1-idx]*scale))
                shape = tuple(shape)
            
            #print '{} to {}'.format(shape, img.shape)
            #img = misc.imresize(img, shape)
            img, label = self.rotate(img, label, angle)
            img, label = self.combine(img,label,center)
            self.imgs.append(img)
            self.labels.append(label)

            if downsample:
                for i, p in enumerate(self.imgs):
                    self.imgs[i] = t.resize(p, (256,256))
                for j, l in enumerate(self.labels):
                    self.labels[j] = t.resize(l, (256,256))
            
            if plot:
                show(img)
                show(label)
         
    
    def save(self, perfix=''):
        #TODO downsample
        from HeartDeepLearning.my_utils import parse_time
        now = parse_time()
        perfix = perfix+'[%d]'%(len(self.imgs))+now
        #os.mkdir(folder)

        imgs = [  i[None,:,:] for i in self.imgs ]
        labels = [ l[None,:,:] for l in self.labels ]
        
        imgs = np.concatenate(imgs, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        import cPickle as pk
        with open(perfix+'.pk','w') as f:
            pk.dump(imgs, f)
            pk.dump(labels, f)

        
if __name__ == '__main__':

    RE = [(50,10),(53,33)]
    g = Maker(RE)
    g.generate(30, 45)
    g.save(downsample=True)


