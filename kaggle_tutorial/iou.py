import dicom, lmdb, cv2, re, sys
import os, fnmatch, shutil, subprocess
import numpy as np
np.random.seed(1234)
import caffe
import pickle as pk

caffe.set_mode_gpu() # or caffe.set_mode_cpu() for machines without a GPU
PK_LIST =  ['o1.pk','o2.pk','o3.pk','v1.pk','v2.pk']
MEAN_VALUE = 77
THRESH = 0.5

def load_net():
    caffe.set_mode_gpu()
    return caffe.Net('fcn_deploy.prototxt', './model_logs/fcn_iter_15000.caffemodel', caffe.TEST)
    

def load_array(pk_name):
    with open(pk_name, 'r') as f:
        img = pk.load(f)
        ll  = pk.load(f)
    return img, ll


def compare_iou(img, label):
    return 2*float(np.sum(img*label))/float(np.sum(label+img))


def compare_series(pre_se, ll_se):
    conj = (pre_se * ll_se).astype('float')
    union = (pre_se + ll_se).astype('float')
    return 2 * conj.sum(axis=(1,2))/(0.05+union.sum(axis=(1,2)))

def sw(img_list):
    if not isinstance(img_list,list):
        img_list = [img_list]
    for img in img_list:
        plt.imshow(img)
        plt.show()

def predit_single(img,net):
    in_ = np.expand_dims(img, axis=0)
    in_ -= np.array([MEAN_VALUE])
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    net.forward()
    prob = net.blobs['prob'].data
    obj = prob[0][1]
    pred = np.where(obj > THRESH, 1, 0)
    return pred


def predit_some(imgs,net):
    imgs = imgs[:,None,:,:]
    imgs -= np.array([MEAN_VALUE])
    net.blobs['data'].reshape(*imgs.shape)
    net.blobs['data'].data[...] = imgs
    net.forward()
    prob = net.blobs['prob'].data
    obj = prob[:,1]
    pred = np.where(obj > THRESH, 1, 0)
    return pred


def predit(im_se,net):
    preds = np.zeros_like(im_se)

    im_se = im_se[:,None,:,:]
    assert im_se.shape[1:] == (1,256,256), 'img_series shape error'

    N = im_se.shape[0]
    # print preds.shape
    for i in range(2):
        print "    Round",i
        im = im_se[N*i/2:N*(i+1)/2]
        net.blobs['data'].reshape(*im.shape)
        net.blobs['data'].data[...] = im
        print 'here'
        net.forward()
        prob = net.blobs['prob'].data
        
        obj = prob[:,1,:,:]
        print obj.shape

        preds[N*i/2:N*(i+1)/2]=np.where(prob[:,1]>THRESH,1,0)
    print "prediction finish"
    return preds

def process_store(pk_name,net):
    img, ll = load_array(pk_name)
    preds = predit_some(img,net)

    assert preds.shape[1:]==(256,256), "preds output shape error"
    iou = compare_series(preds,ll)

    csv_name = pk_name.split(".")[0]+".csv"
    np.savetxt(csv_name,iou,delimiter="\n")

    with open(pk_name.split(".")[0]+"_pred.pk",'w') as f:
        pk.dump(preds, f)

    mean = float(iou.sum())/float(iou.shape[0])
    std = float(((iou - mean)**2).sum())/ float(iou.shape[0])
    std = np.sqrt(std)
    print "%s mean: %d, std: %d"%(pk_name.split(".")[0], mean, std)
    return iou

if __name__ == '__main__':

    net = load_net()

    for pk_name in PK_LIST:
        process_store(pk_name,net)

    # print "online mean, std >", o_iou.mean, o_iou.std
    # print "validate mean, s>>", v_iou.mean, v_iou.std
    

