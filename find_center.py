import numpy as np
from PIL import Image,ImageDraw;
from skimage import exposure;

def locate(imgs):
    """
    find the approximate center of the left ventricle and a bounding box of it.
    Imgs have the shape of [N,1,256,256]
    """
    
    p10,p90 = np.percentile(imgs,(10,90));
    imgs = exposure.rescale_intensity(imgs, in_range=(p10, p90));
    imgs = imgs/imgs.max();
    x=np.std(imgs,axis=0);
    print 'X', x.shape, x
    print x.mean()
    
    lx, ly = imgs[0].shape;
    xm,ym,delta = lx//2,ly//2,min(lx*2//5,ly*2//5);
    cut = np.percentile(x,95);
    img_tmp = Image.new('L', x.shape[::-1], 0)
    ImageDraw.Draw(img_tmp).ellipse((ym-delta,xm-delta,ym+delta,xm+delta), outline=1, fill=1)
    mask = np.array(img_tmp);
    y=(x>cut) & (mask>0.5);
    xm,ym = np.where(y);
    delta = (np.std(xm)+np.std(ym));
    xm = np.mean(xm);
    ym = np.mean(ym);
    delta = 1.2*delta;
    delta = min(delta,xm,ym,lx-xm,ly-ym);
    print xm, ym, delta

    img_tmp = Image.new('L', x.shape[::-1], 0)
    ImageDraw.Draw(img_tmp).ellipse((ym-delta,xm-delta,ym+delta,xm+delta), outline=1, fill=1)
    mask = np.array(img_tmp);
    y=(x>cut) & (mask>0.5);
    xm,ym = np.where(y);
    delta = (np.std(xm)+np.std(ym));
    xm = np.mean(xm);
    #delta = max(50/np.sqrt(self.area_multiplier[Nsaxmid]),delta);
    ym = np.mean(ym)+delta/7;
    #move to right a little bit if it is dark a the point
    xm,ym,delta = int(xm),int(ym),int(delta);
    xms = [(xm,ym),(xm,ym+6),(xm+6,ym+6),(xm-6,ym+6)];
    N = len(xms);
    b = np.zeros(N);
    for i in range(N):
        b[i] = max(np.mean(imgs[:,xms[i][0]-4:xms[i][0]+4,xms[i][1]-4:xms[i][1]+4]),\
            np.mean(imgs[:,xms[i][0]-4:xms[i][0]+4,xms[i][1]-4:xms[i][1]+4]));
    i = np.argmax(b);
    xm,ym = xms[i];

    delta = delta*1.6;
    xm,ym,delta = int(xm),int(ym),int(delta);
    delta = min(delta,xm,ym,lx-xm,ly-ym);
    return xm,ym,delta;


if __name__ == '__main__':
    import pickle as pk

    with open('DATA/small/o1.pk','r') as f:
        imgs = pk.load(f)
        labels = pk.load(f)

    xm, ym, delta = locate(imgs)
    print xm, ym, delta