import dicom, lmdb, cv2, re, sys
import os, fnmatch, shutil, subprocess
import numpy as np
np.random.seed(1234)
import caffe
import pickle as pk

caffe.set_mode_gpu() # or caffe.set_mode_cpu() for machines without a GPU
PK_NAME =  ['online.pk','validate.pk']
MEAN_VALUE = 77
THRESH = 0.5

def load_net():
    caffe.set_mode_gpu()
    return caffe.Net('fcn_deploy.prototxt', './model_logs/fcn_iter_15000.caffemodel', caffe.TEST)
    

def load_array(pk_names):
    with open("online.pk", 'r') as f:
        o_img = pk.load(f)
        o_ll  = pk.load(f)

    with open("validate.pk",'r') as f:
        v_img = pk.load(f)
        v_ll  = pk.load(f)
    return o_img, o_ll, v_img, v_ll


def compare_iou(img, label):
    return 2*float(np.sum(img*label))/float(np.sum(label+img))


def compare_series(pre_se, ll_se):
    conj = (pre_se * ll_se).astype('float')
    union = (pre_se + ll_se).astype('float')
    return 2 * conj.sum(axis=(1,2))/union.sum(axis=(1,2))


def predit(net,im_se):
    im_se = im_se[:,None,:,:]
    assert im_se.shape[1:] == (1,256,256), 'img_series shape error'

    N = im_se.shape[0]
    preds = np.zeros_like(im_se)
    for i in range(1):
        print "    Round",i
        im_se = im_se[N/2*i:N/2*(i+1)]
        net.blobs['data'].reshape(*im_se.shape)
        net.blobs['data'].data[...] = im_se
        net.forward()
        prob = net.blobs['prob'].data
        preds[N/2*i:N/2*(i+1)]=np.where(prob[0][1]>THRESH,1,0)
    return preds


if __name__ == '__main__':

    o_img, o_ll, v_img, v_ll = load_array(PK_NAME)
    net = load_net()
    print "Done loading"

#    if True:
#        print "cut data in half"
#        N = o_img.shape[0]
#        o_img = o_img[:N/2]
#        o_ll = o_ll[:N/2]

    print "processing online .."
    o_pred = predit(net,o_img)
    o_iou = compare_series(o_pred,o_ll)
    np.savetxt('o_iou.csv',o_iou,delimiter="\n")

    print "process validate .."
    v_pred = predit(net,v_img)
    v_iou = compare_series(v_pred,v_ll)
    np.savetxt('v_iou.csv',v_iou,delimiter="\n")

    print "online mean, std >>", o_iou.mean, o_iou.std
    print "validate mean, std >>", v_iou.mean, v_iou.std
    

